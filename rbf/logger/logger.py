import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
import shutil

from rbf.utils.image_utils import save_tensor
from rbf.utils.print_utils import print_info
from rbf.utils.extra_utils import ignore_kwargs
from rbf import shared_modules


class BaseLogger(ABC):
    """
    A simple abstract logger class for logging
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, camera, images) -> None:
        pass

    def get_extra_cameras(self, step):
        return []

    def end_logging(self):
        pass


class SelfLogger(BaseLogger):
    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        log_interval: int = 100
        use_encoder_decoder: bool = False

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.root_dir = self.cfg.root_dir
        self.training_dir = os.path.join(self.cfg.root_dir, "training")
        self.debug_dir = os.path.join(self.cfg.root_dir, f"debug")

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # print_info(f"SelfLogger initialized.")
        print_info(f"Logging results to {self.root_dir}") if not shared_modules.OFF_LOG else None

    def __call__(self, step, camera, images) -> None:
        if step % self.cfg.log_interval != 0:
            return
        images = shared_modules.model.render_self()
        if self.cfg.use_encoder_decoder:
            latents = shared_modules.prior.encode_image_if_needed(images)
            images = shared_modules.prior.decode_latent(latents)
        images.clip_(0, 1)
        save_tensor(
            images,
            os.path.join(self.training_dir, f"training_{step:05d}.png"),
            save_type="cat_image",
        )


    def end_logging(self):
        num_files = len(os.listdir(self.training_dir))

        if num_files == 1:
            shutil.copyfile(
                os.path.join(self.training_dir, "training_00000_0.png"),
                os.path.join(self.root_dir, "result.png"),
            )
        else:
            if num_files < 20:  # lerp between 1 and 4fps for 2-20 files
                fps = int(2 + (num_files - 2) * 3 / 18)
            elif num_files < 100:  # lerp between 4 and 20fps for 20-100 files
                fps = int(4 + (num_files - 20) * 16 / 80)
            elif num_files < 1000:  # lerp between 20 and 30fps for 100-1000 files
                fps = int(20 + (num_files - 100) * 10 / 900)
            else:  # 30fps for 1000+ files
                fps = 30

            # convert_to_video(
            #     self.training_dir,
            #     os.path.join(self.root_dir, "result.mp4"),
            #     fps=fps,
            #     force=True,
            # )


    def log_potentials(self, potential):
        _particles, _steps = potential.shape
        with open(os.path.join(self.root_dir, "potential.txt"), "a") as f:
            for _p in range(_particles):    
                for _s in range(_steps):
                    f.write(f"{potential[_p][_s]} ")

            f.write(f"{potential[-1][0]} ")
