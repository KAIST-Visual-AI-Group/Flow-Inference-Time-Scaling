from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from rbf import shared_modules as sm 

from rbf.utils.image_utils import save_tensor, pil_to_torch
from rbf.utils.print_utils import print_info
from rbf.utils.extra_utils import ignore_kwargs
from rbf import shared_modules

from rbf.model.base import BaseModel

class FluxImageModel(BaseModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        max_steps: int = 10000
        device: int = 0
        width: int = 1024
        height: int = 1024
        data_dim: int = 64
        initialization: str = "random"  # random, zero, gray, image
        init_img_path: Optional[str] = None
        channels: int = 4096
        batch_size: int = 1

        learning_rate: float = 0.0

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None
        self.image_latest = None

    @torch.no_grad()
    def load(self, path: str) -> None:
        img = Image.open(path).convert("RGB")
        img = pil_to_torch(img).to(self.cfg.device)

        if self.cfg.channels == 3:
            return img
        
        elif self.cfg.channels == 4096:
            latent = shared_modules.prior.encode_image(img)
            return latent
        
        else:
            raise ValueError(f"Channels must be 3 or 4096, got {self.cfg.channels}")


    @torch.no_grad()
    def save(self, path: str) -> None:
        image = self.render_self()
        save_tensor(image, path)
        
        if self.image_latest is not None:
            self.image = self.image_latest
            image_latest = self.render_self()
            save_tensor(image_latest, path+"_latest")

    def initialize_image(
        self, C, H, W, D, init_method="random", img_path=None
    ) -> torch.Tensor:
        if C == 3:
            print_info("Detected 3 channels. Assuming RGB-space image.") if not sm.OFF_LOG else None
            if init_method == "random":
                image = torch.rand(self.cfg.batch_size, 3, H, W, device=self.cfg.device)
            elif init_method == "zero":
                image = torch.zeros(self.cfg.batch_size, 3, H, W, device=self.cfg.device)
            elif init_method == "gray":
                image = torch.full((self.cfg.batch_size, 3, H, W), 0.5, device=self.cfg.device)
            elif init_method == "image":
                print_info(f"Loading image from {img_path}") if not sm.OFF_LOG else None
                # image = Image.open(img_path).convert("RGB")
                # image = pil_to_torch(image).to(self.cfg.device).squeeze(0)
                image = self.load(img_path)
            else:
                raise ValueError(f"Invalid initialization: {init_method}")
            
        else:
            print_info(f"Detected {C} channels. Assuming latent-space image.") if not sm.OFF_LOG else None
            if init_method == "random":
                image = torch.randn(self.cfg.batch_size, 4096, D, device=self.cfg.device)
            elif init_method == "zero_latent":
                image = torch.zeros(self.cfg.batch_size, 4096, D, device=self.cfg.device)
            elif init_method == "zero":
                S = int(shared_modules.prior.pipeline.vae_scale_factor)
                image = torch.zeros(self.cfg.batch_size, 3, H * S, W * S, device=self.cfg.device)
                image = shared_modules.prior.encode_image(image)
            elif init_method == "gray":
                S = int(shared_modules.prior.pipeline.vae_scale_factor)
                image = torch.full((self.cfg.batch_size, 3, H * S, W * S), 0.5, device=self.cfg.device)
                image = shared_modules.prior.encode_image(image)
            elif init_method == "image":
                print_info(f"Loading image from {img_path}") if not sm.OFF_LOG else None
                # image = Image.open(img_path).convert("RGB")
                # image = pil_to_torch(image).to(self.cfg.device)
                # image = shared_modules.prior.encode_image(image).squeeze(0)
                image = self.load(img_path)
            else:
                raise ValueError(f"Invalid initialization: {init_method}")

        return image


    def prepare_optimization(self) -> None:
        self.image = torch.nn.Parameter(
            self.initialize_image(
                self.cfg.channels,
                self.cfg.height,
                self.cfg.width,
                self.cfg.data_dim,
                self.cfg.initialization,
                self.cfg.init_img_path,
            )
        )
        self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)


    def render(self, camera) -> torch.Tensor:
        assert 1==2, "This function is not used"

        tf = camera.get("transforms", lambda x: x)
        img = tf(self.image)
        img_resized = F.interpolate(
            img,
            size=(camera["height"], camera["width"]),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "image": img_resized,
            "alpha": torch.ones(
                1, 1, self.cfg.height, self.cfg.width, device=self.cfg.device
            ),
        }

    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        if 3 in self.image.shape:
            image = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)

        else:
            # [C D] -> [B C D] 
            image = self.image if self.image.dim() == 3 else self.image.unsqueeze(0) 

        if image.shape[1] == 3:
            pass
            # latent = shared_modules.prior.encode_image(image)
            # image = shared_modules.prior.decode_latent(latent)
        # elif image.shape[1] == 4096:
        else:
            image = shared_modules.prior.decode_latent(image)

        return image

    def optimize(self, step: int) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()
        if hasattr(self, "scheduler"):
            last_lr = self.scheduler.get_last_lr()
            self.scheduler.step()
            print_info(f"Using learning scheduler at step {step}: {last_lr} -> {self.scheduler.get_last_lr()}") if not sm.OFF_LOG else None

    def closed_form_optimize(self, step, camera, target):
        if self.image.shape[0] == 3:
            target = shared_modules.prior.decode_latent_if_needed(target)
        elif self.image.shape[0] == 4:
            target = shared_modules.prior.encode_image_if_needed(target)

        self.image = target

    def regularize(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.cfg.device)
        image = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)
        if image.shape[1] == 3:
            with torch.no_grad():
                latent = shared_modules.prior.encode_image(image)
                gt_image = shared_modules.prior.decode_latent(latent)
            recon_loss = 0.05 * F.mse_loss(image, gt_image)
            print(recon_loss.item())
            return recon_loss
        elif image.shape[1] == 4:
            return torch.tensor(0.0, device=self.cfg.device)

    @torch.no_grad()
    def render_eval(self, path) -> torch.Tensor:
        pass 


    def guide_x0(self, step, target):
        self.image = target
        return target