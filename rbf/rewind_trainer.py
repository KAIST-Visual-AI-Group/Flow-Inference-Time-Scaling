import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import math 
from PIL import Image
from tqdm import tqdm

from rbf import shared_modules as sm
from rbf.prior import PRIORs
from rbf.logger import LOGGERs
from rbf.model import MODELs
from rbf.corrector import CORRECTORs
from rbf.utils.extra_utils import (
    ignore_kwargs,
    get_class_filename,
    redirect_stdout_to_tqdm,
)
from rbf.utils.extra_utils import redirected_trange as re_trange
from rbf.utils.print_utils import print_with_box, print_info, print_warning, print_note
from rbf.utils.image_utils import torch_to_pil_batch, image_grid, torch_to_pil


class RewindTrainer:
    """
    Abstract base class for all trainers.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        output: str = "output"
        device: int = 0

        model: str = ""
        prior: str = ""
        logger: str = "simple"
        corrector: str = "ddim"
        

        batch_size: int = 1
        width: int = 1024
        height: int = 1024
        t_max: int = 1000

        init_step: int = 0
        max_steps: int = 10
        
        save_source: bool = False
        disable_debug: bool = False

        log_interval: int = 5
        sample_method: str = None
        diffusion_norm: str = None
        n_max_log: int = 10

        # Sampling-based approaches framework
        n_particles: int = 1
        tau_norm: float = 0.0
        filtering_method: str = None

        # SoP config 
        forward_step: float = 780.0
        backward_step: float = 810.0
        ode_seed_t: float = 120.0

        img_idx: int = 0
        benchmark: bool = False


    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        sm.model = MODELs[self.cfg.model](cfg_dict)
        sm.prior = PRIORs[self.cfg.prior](cfg_dict)
        sm.logger = LOGGERs[self.cfg.logger](cfg_dict)
        sm.corrector = CORRECTORs[self.cfg.corrector](cfg_dict)

        self.eval_dir = os.path.join(self.cfg.root_dir, "eval")
        os.makedirs(self.cfg.root_dir, exist_ok=True)
        os.makedirs(f"{self.cfg.root_dir}/debug", exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

        if self.cfg.save_source:
            os.makedirs(f"{self.cfg.root_dir}/src", exist_ok=True)
            for module in [
                sm.model,
                sm.prior,
                sm.logger,
                sm.corrector,
            ]:
                filename = get_class_filename(module)
                os.system(f"cp {filename} {self.cfg.root_dir}/src/")

            from .prior.base import Prior

            filename = get_class_filename(Prior)
            os.system(f"cp {filename} {self.cfg.root_dir}/src/base_prior.py")

        sm.model.prepare_optimization()
        
        assert self.cfg.sample_method is not None, "sample_method must be provided"
        if self.cfg.sample_method == "sde":
            assert self.cfg.diffusion_norm > 0, "diffusion_norm must be provided for SDE-based methods"


    def train_single_step(self, sample_dict: dict) -> Any:
        with torch.no_grad():
            step = sample_dict["step"]
            # pbar = sample_dict["pbar"]

            if step == 0:
                # Flow-based models
                if self.cfg.prior in ["flux", "instaflow"]:
                    latent_noisy = sm.prior.init_latent(
                        self.cfg.batch_size
                    ) # B, 4, H, W (x_T)
                    
                else:
                    # Stable Diffusion 
                    init_shape = (self.cfg.batch_size, 4, self.cfg.height, self.cfg.width)
                    latent_noisy = torch.randn(
                        init_shape, dtype=sm.prior.dtype, device=sm.prior.device,
                    )

                # ==============================================================
                # 0. Initial seed latent
                # ==============================================================
                latent_noisy = sm.prior.solve_ode(
                    latent_noisy, 
                    target_t=float(self.cfg.ode_seed_t),
                    return_tweedie=False,
                ) # B, 4, H, W

                backward_target_t = torch.full(
                    (self.cfg.batch_size * self.cfg.n_particles,), 
                    self.cfg.ode_seed_t
                ).to(sm.prior.device).view(-1, 1, 1)

                forward_target_t = torch.full(
                    (self.cfg.batch_size * self.cfg.n_particles,), 
                    self.cfg.ode_seed_t + self.cfg.forward_step
                ).to(sm.prior.device).view(-1, 1, 1) # BN / forward t

            else:
                latent_noisy = sample_dict["latent_noisy"]
                forward_target_t = sample_dict["forward_target_t"]
                backward_target_t = sample_dict["backward_target_t"]

            assert latent_noisy.shape[0] == self.cfg.batch_size, f"{latent_noisy.shape[0]} != {self.cfg.batch_size}"

            
            # ==============================================================
            # 1. Add noise to x_forward_target_t
            # ==============================================================
            # TODO: Replace with Markovian kernel
            print("Adding noise from ", backward_target_t[0].item(), "to", forward_target_t[0].item()) if not sm.OFF_LOG else None
            # scheduler_output = sm.prior.new_scheduler(forward_target_t.float() / 1000.0)
            # alpha_t = scheduler_output.alpha_t.to(latent_noisy.dtype)
            # sigma_t = scheduler_output.sigma_t.to(latent_noisy.dtype)

            # noise = torch.randn(
            #     (self.cfg.batch_size * self.cfg.n_particles, *(latent_noisy.shape[1:])), 
            #     device=latent_noisy.device, 
            #     dtype=latent_noisy.dtype
            # )

            # bn_latent_noisy = torch.repeat_interleave(latent_noisy, self.cfg.n_particles, dim=0) # BN, 4, H, W
            # print("latent_noisy", latent_noisy.shape, "bn_latent_noisy", bn_latent_noisy.shape) if not sm.OFF_LOG else None
            # assert bn_latent_noisy.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"Batch size should be same as the input: got {bn_latent_noisy.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"

            # latent_noisy = alpha_t * bn_latent_noisy + sigma_t * noise # BN, 4, H, W (Forward T)
            # assert latent_noisy.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"Batch size should be same as the input: got {latent_noisy.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"

            latent_noisy = torch.repeat_interleave(latent_noisy, self.cfg.n_particles, dim=0) # BN, 4, H, W

            forward_target_coeff = sm.prior.new_scheduler(forward_target_t / 1000.0)
            sigma_target = forward_target_coeff.sigma_t
            alpha_target = forward_target_coeff.alpha_t

            cur_coeff = sm.prior.new_scheduler(backward_target_t / 1000.0)
            sigma_cur = cur_coeff.sigma_t
            alpha_cur = cur_coeff.alpha_t

            assert alpha_cur[0].item() > 0, f"alpha_cur should be positive. Got {alpha_cur}"

            alpha_t_alpha_s = alpha_target / alpha_cur
            mean = alpha_t_alpha_s * latent_noisy
            var = sigma_target ** 2 - alpha_t_alpha_s ** 2 * sigma_cur ** 2

            latent_noisy = (mean + torch.randn_like(mean) * var.sqrt()).to(sm.prior.dtype)

            # ==============================================================
            # 2. Solve ODE to backward_target_t
            # ==============================================================
            backward_target_t = torch.clamp(forward_target_t - self.cfg.backward_step, min=0.0) # BN / backward t
            print("Solving ODE from ", forward_target_t[0].item(), "to", backward_target_t[0].item()) if not sm.OFF_LOG else None

            latent_noisy, tweedie = sm.prior.solve_ode(
                latent_noisy, 
                target_t=backward_target_t,
                source_t=forward_target_t,
                return_tweedie=True,
            )

            assert latent_noisy.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"Batch size should be same as the input: got {latent_noisy.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"
            assert tweedie.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"Batch size should be same as the input: got {tweedie.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"

            latent_noisy, tweedie, _ = sm.corrector.post_correct(
                latent_noisy, 
                tweedie, 
                torch.zeros_like(latent_noisy), 
                step,
            )  # B, 4, H, W

            assert latent_noisy.shape[0] == self.cfg.batch_size, f"{latent_noisy.shape[0]} != {self.cfg.batch_size}"
            assert tweedie.shape[0] == self.cfg.batch_size, f"{tweedie.shape[0]} != {self.cfg.batch_size}"

            forward_target_t = (backward_target_t + self.cfg.forward_step)

            sample_dict = {
                "forward_target_t": forward_target_t,
                "backward_target_t": backward_target_t,
                "latent_noisy": latent_noisy,
            }
            # pbar.update(1)

            tweedie = sm.model.guide_x0(
                step, tweedie
            ) # B, 4, H, W

            # ===============================================
            # Logging
            # ===============================================
            if (not self.cfg.disable_debug) and (step % self.cfg.log_interval == 0) and not sm.DO_NOT_SAVE_INTERMEDIATE_IMAGES:
                # NOTE: Batch-wise logging
                print_info(f"Logging at {step}. Step: {step}")

                xts_logs = []
                x0s_logs = []
                for _b in range(len(latent_noisy[:min(len(latent_noisy), self.cfg.n_max_log)])):
                    prev_latent_for_log = latent_noisy[_b:_b+1]
                    tweedie_for_log = tweedie[_b:_b+1]

                    if prev_latent_for_log.shape[1] != 3:
                        prev_latent_for_log = sm.prior.decode_latent(prev_latent_for_log)
                    
                    if tweedie_for_log.shape[1] != 3:
                        tweedie_for_log = sm.prior.decode_latent(tweedie_for_log)

                    xts_logs.append(prev_latent_for_log)
                    x0s_logs.append(tweedie_for_log)

                pil_x0s = torch_to_pil_batch(
                    torch.cat(x0s_logs), 
                    is_grayscale=False,
                )
                grid_x0 = image_grid(
                    pil_x0s, 1, len(pil_x0s)).resize((256 * len(pil_x0s), 256)
                )
                
                pil_xts = torch_to_pil_batch(
                    torch.cat(xts_logs), 
                    is_grayscale=False
                )
                
                grid_xt = image_grid(
                    pil_xts, 1, len(pil_xts)).resize((256 * len(pil_xts), 256)
                )

                # Merge xts/x0s grids
                grid_img = image_grid(
                    [grid_x0, grid_xt], 2, 1
                )

                grid_img.save(
                    os.path.join(sm.logger.training_dir, f"training_{step:03d}_{int(step)}.png")
                )

        
            # ===============================================
            # Pick final sample 
            # ===============================================
            if (torch.all(backward_target_t <= 1e-6).item() and self.cfg.batch_size > 1):
                # Pick the proposed sample with the highest reward
                tweedie = sm.corrector.final_correct(tweedie, step)
                sm.model.image = tweedie

                with open(os.path.join(sm.logger.debug_dir, "nfe.txt"), "w") as f:
                    f.write(f"{sm.prior.nfe}\n")

                

        return sample_dict


    def train(self):
        sample_dict = dict()

        _step = math.ceil(self.cfg.ode_seed_t / (self.cfg.backward_step - self.cfg.forward_step))
        print("Total steps", _step) if not sm.OFF_LOG else None
        
        if not self.cfg.benchmark:
            for _s in range(_step):
                sample_dict["step"] = _s
                sample_dict = self.train_single_step(sample_dict)
                
            backward_target_t = sample_dict["backward_target_t"]
            assert torch.all(backward_target_t <= 1e-6).item(), "Backward target t should be zero"

            output_filename = os.path.join(
                self.cfg.root_dir, self.cfg.output
            )
            sm.model.save(output_filename)
            if hasattr(sm.model, "render_eval"):
                print_info("render_eval detected. Rendering the final image...")
                sm.model.render_eval(self.eval_dir)

            return output_filename
        
        else:
            for _s in tqdm(range(_step), desc="Denoising Step", total=_step, dynamic_ncols=True, leave=False):
                sample_dict["step"] = _s
                sample_dict = self.train_single_step(sample_dict)
                
            backward_target_t = sample_dict["backward_target_t"]
            assert torch.all(backward_target_t <= 1e-6).item(), "Backward target t should be zero"

            output_filename = os.path.join(self.cfg.root_dir, "generated_images", f"{self.cfg.img_idx:05d}.png")
            final = sm.model.image
            final_img = sm.prior.decode_latent(final)
            final_img = final_img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            final_img = (final_img*255).astype('uint8')
            Image.fromarray(final_img).save(output_filename)