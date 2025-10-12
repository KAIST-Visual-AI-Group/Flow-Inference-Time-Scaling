import os
from dataclasses import dataclass
from typing import Any, Optional
from tqdm import tqdm
from PIL import Image

import torch
from rbf import shared_modules as sm
from rbf.prior import PRIORs
from rbf.logger import LOGGERs
from rbf.model import MODELs
from rbf.corrector import CORRECTORs
from rbf.time_sampler import TIME_SAMPLERs
from rbf.utils.extra_utils import (
    ignore_kwargs,
    get_class_filename,
    redirect_stdout_to_tqdm,
)
from rbf.utils.extra_utils import redirected_trange as re_trange
from rbf.utils.print_utils import print_with_box, print_info, print_warning, print_note
from rbf.utils.image_utils import torch_to_pil_batch, image_grid


class SVDDDPS:
    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        output: str = "output"
        device: int = 0
        seed: int = 0

        model: str = ""
        prior: str = ""
        logger: str = "simple"
        corrector: str = "ddim"
        time_sampler: str = "flux_scheduler"
        
        batch_size: int = 1
        width: int = 512
        height: int = 512
        t_max: int = 1000

        max_steps: int = 30
        init_step: int = 0
        
        save_source: bool = False
        disable_debug: bool = False

        log_interval: int = 5
        sample_method: str = None
        diffusion_norm: str = None
        n_max_log: int = 10

        # Sampling-based approaches framework
        block_size: int = 1
        n_particles: int = 1
        filtering_method: str = None

        img_idx: int = 0
        benchmark: bool = False
        prompt: str = None

        # Guidance param
        guidance_strength: float = 0.01
        minibatch_size: int = 1

        # DAS
        tempering_gamma: float = 0.008

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        sm.model = MODELs[self.cfg.model](cfg_dict)
        sm.time_sampler = TIME_SAMPLERs[self.cfg.time_sampler](cfg_dict)
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
        with torch.enable_grad():
            step = sample_dict["step"]
            model_pred = sample_dict.get("model_pred", None)
            pbar = sample_dict["pbar"]
            reward_tensor = sample_dict.get("reward_tensor", None)

            t_curr, d_t = sm.time_sampler(step)
            derivative = sample_dict.get("derivative", None)

            if step == 0:
                # Flow-based models
                if self.cfg.prior in ["flux", "instaflow", "flux_fill", "sd2", "sd"]:
                    latent_noisy = sm.prior.init_latent(
                        self.cfg.batch_size
                    ) # B, 4, H, W (x_T)
                else:
                    # Stable Diffusion 
                    init_shape = (self.cfg.batch_size, 4, self.cfg.height, self.cfg.width)
                    latent_noisy = torch.randn(
                        init_shape, dtype=sm.prior.dtype, device=sm.prior.device,
                    )

            else:
                latent_noisy = sample_dict["xts"]
                assert latent_noisy.shape[0] == self.cfg.batch_size, f"{latent_noisy.shape[0]} != {self.cfg.batch_size}"
            latent_noisy = torch.repeat_interleave(latent_noisy, self.cfg.n_particles, dim=0) # B*N, 4, H, W

            if model_pred is None:
                with torch.no_grad():
                    assert step == 0, "model_pred must be provided for step > 0"
                    model_pred = sm.prior.compute_velocity_transform_scheduler(
                        latent_noisy, 
                        t_curr,
                    ) # B*N, 4, H, W (v_T)
                    tweedie = sm.prior.get_tweedie(
                        latent_noisy,
                        model_pred,
                        t_curr,
                    ) # B*N, 4, H, W (x_0|T)
                derivative = torch.zeros_like(latent_noisy)
                
            else:
                model_pred = torch.repeat_interleave(model_pred, self.cfg.n_particles, dim=0) # B*N, 4, H, W
                tweedie = sample_dict["tweedie"]

            assert latent_noisy.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"{latent_noisy.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"
            assert model_pred.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"{model_pred.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"
            
            derivative_list = list()
            reward_list = list()
            model_pred_list = list()
            tweedie_list = list()

            for _ in range(self.cfg.block_size):
                t_prev = torch.clamp(t_curr - d_t, min=0)

                latent_noisy = sm.prior.step(
                    latent_noisy, # B*N, 4, H, W (x_t)
                    t_curr=t_curr.to(latent_noisy), 
                    d_t=d_t.to(latent_noisy) / 1000.0,
                    model_pred=model_pred, # B*N, 4, H, W (v_t)
                    prev_timestep=t_prev, # logging purpose
                ) # B*N, 4, H, W (x_t-1)
                pbar.update(1)

                assert model_pred.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"{model_pred.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"
                assert latent_noisy.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"{latent_noisy.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"

                tempering_factor = torch.tensor([min((1 + self.cfg.tempering_gamma) ** (step + 1) - 1, 1.)], device=self.cfg.device, dtype=torch.bfloat16)
                derivative = derivative * tempering_factor
                t_curr_coeff = sm.prior.new_scheduler(t=(t_curr/1000.0))
                alpha_t = t_curr_coeff.alpha_t.to(torch.bfloat16)

                latent_noisy = latent_noisy + self.cfg.guidance_strength * derivative * alpha_t

                step += 1
                if step == self.cfg.max_steps:
                    assert torch.all(t_prev <= 1e-6).item(), f"{t_prev} not close to 0"
                    break

                for _idx in range(latent_noisy.shape[0]):
                    cur_latent_noisy = latent_noisy[_idx:_idx+1].clone().detach()
                    cur_latent_noisy.requires_grad_(True)
                    t_curr, d_t = sm.time_sampler(step)
                    t_curr = t_curr[0:1, ...]
                    cur_model_pred = sm.prior.compute_velocity_transform_scheduler(
                        cur_latent_noisy, # x_t-1
                        t_curr, # t-1
                    ) # B*N, 4, H, W (v_t-1)
                    cur_tweedie = sm.prior.get_tweedie(
                        cur_latent_noisy, # x_t-1 / x_t-1+tau
                        cur_model_pred, # v_t-1
                        t_curr, # t-1
                    ) # B*N, 4, H, W (x_0|t-1)

                    target = list()
                    decoded_tweedie_list = list()
                    for __i in range(0, len(cur_tweedie), self.cfg.minibatch_size):
                        cur_batch_size = min(self.cfg.minibatch_size, len(cur_tweedie) - __i)
                        decoded_tweedies = sm.prior.decode_latent(cur_tweedie[__i:__i+cur_batch_size])
                        decoded_tweedie_list.append(decoded_tweedies.clone().detach())
                        target += [sm.corrector.reward_model.preprocess(decoded_tweedies)]

                    target = torch.cat(target)
                    decoded_tweedies_list = torch.cat(decoded_tweedie_list)
                    new_rewards = sm.corrector.reward_model(target, step, decoded_tweedies_list)

                    reward_list.append(new_rewards.item())
                    model_pred_list.append(cur_model_pred.clone().detach())
                    tweedie_list.append(cur_tweedie.clone().detach())

                    new_rewards = new_rewards * 200.0

                    cur_derivative = torch.autograd.grad(new_rewards, cur_latent_noisy, torch.ones_like(new_rewards))[0]
                    derivative_list.append(cur_derivative.clone().detach())

                model_pred = torch.cat(model_pred_list)
                derivative = torch.cat(derivative_list)

            if step != self.cfg.max_steps:
                tweedie = torch.cat(tweedie_list)
                reward_tensor = torch.tensor(reward_list)
            log_tweedie = tweedie.clone()
            assert tweedie.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"{tweedie.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"

            reward_tensor = reward_tensor.reshape(self.cfg.batch_size, self.cfg.n_particles)
            best_idx = torch.argmax(reward_tensor, dim=1)
            t_arg = torch.arange(self.cfg.batch_size) * self.cfg.n_particles + best_idx

            if step != self.cfg.max_steps:
                model_pred = model_pred[t_arg]
                derivative = derivative[t_arg]
            latent_noisy = latent_noisy[t_arg]
            tweedie = tweedie[t_arg]
            tweedie = sm.model.guide_x0(step, tweedie) # B, 4, H, W

            sample_dict = {
                "xts": latent_noisy.detach(),
                "model_pred": model_pred.detach(),
                "tweedie": log_tweedie.detach(),
                "step": step,
                "pbar": pbar,
                "derivative": derivative.detach(),
                "reward_tensor": reward_tensor,
            }

            # ===============================================
            # Logging
            # ===============================================
            if (not self.cfg.disable_debug) and (step % self.cfg.log_interval == 0) and not sm.DO_NOT_SAVE_INTERMEDIATE_IMAGES:
                LOG_RESOLUTION_DOWNSCALE = 4
                LOG_H = self.cfg.height // LOG_RESOLUTION_DOWNSCALE
                LOG_W = self.cfg.width // LOG_RESOLUTION_DOWNSCALE
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

                pil_x0s = torch_to_pil_batch(torch.cat(x0s_logs), is_grayscale=False,)
                grid_x0 = image_grid(pil_x0s, 1, len(pil_x0s)).resize((LOG_W * len(pil_x0s), LOG_H))
                
                pil_xts = torch_to_pil_batch(torch.cat(xts_logs), is_grayscale=False)
                grid_xt = image_grid(pil_xts, 1, len(pil_xts)).resize((LOG_W * len(pil_xts), LOG_H))

                grid_img = image_grid([grid_x0, grid_xt], 2, 1)
                grid_img.save(os.path.join(sm.logger.training_dir, f"training_{step:03d}_{int(step-1)}.png"))

        
            # ===============================================
            # Pick final sample 
            # ===============================================
            if (step == self.cfg.max_steps and self.cfg.batch_size > 1):
                tweedie = sm.corrector.final_correct(tweedie, step)
                sm.model.image = tweedie
        return sample_dict


    def train(self):
        sample_dict = {"step": 0}
        if not self.cfg.benchmark:
            with redirect_stdout_to_tqdm():
                with re_trange(
                    self.cfg.init_step, 
                    self.cfg.max_steps, 
                    self.cfg.block_size,
                    position=0, desc="Denoising Step", 
                    initial=self.cfg.init_step, 
                    total=self.cfg.max_steps,) as pbar:

                    sample_dict["pbar"] = pbar
                    for _ in range((self.cfg.max_steps + self.cfg.block_size - 1) // self.cfg.block_size):
                        sample_dict = self.train_single_step(sample_dict)

                    pbar.close()
                output_filename = os.path.join(
                    self.cfg.root_dir, self.cfg.output
                )
                sm.model.save(output_filename)
                if hasattr(sm.model, "render_eval"):
                    print_info("render_eval detected. Rendering the final image...") if not sm.OFF_LOG else None
                    sm.model.render_eval(self.eval_dir)
                return output_filename
        else:
            with tqdm(range(self.cfg.init_step, self.cfg.max_steps, self.cfg.block_size), desc="Denoising Step", total=self.cfg.max_steps, dynamic_ncols=True, leave=False) as pbar:
                sample_dict["pbar"] = pbar
                for _ in range((self.cfg.max_steps + self.cfg.block_size - 1) // self.cfg.block_size):
                    sample_dict = self.train_single_step(sample_dict)
                pbar.close()

            output_filename = os.path.join(self.cfg.root_dir, f"{self.cfg.prompt}_{self.cfg.seed:05d}.png")
            final = sm.model.image
            final_img = sm.prior.decode_latent(final)
            final_img = final_img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            final_img = (final_img*255).astype('uint8')
            Image.fromarray(final_img).save(output_filename)
