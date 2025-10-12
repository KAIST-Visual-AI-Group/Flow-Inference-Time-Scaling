import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image 
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import torchvision
from functorch import grad, vmap

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


reward_list = dict()

class DPSTrainer:
    """
    Abstract base class for all trainers.
    """

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
        max_nfe: int = 100

        max_steps: int = 100
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
        tau_norm: float = 0.0
        filtering_method: str = None

        img_idx: int = 0
        benchmark: bool = False
        prompt: str = None

        # Guidance param
        guidance_strength: float = 1.0
        minibatch_size: int = 5

        # DAS
        tempering_gamma: float = 0.008


    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        sm.model = MODELs[self.cfg.model](cfg_dict)
        sm.time_sampler = TIME_SAMPLERs[self.cfg.time_sampler](cfg_dict)
        sm.prior = PRIORs[self.cfg.prior](cfg_dict)
        sm.logger = LOGGERs[self.cfg.logger](cfg_dict)
        sm.corrector = CORRECTORs[self.cfg.corrector](cfg_dict)
        sm.time_sampler = TIME_SAMPLERs[self.cfg.time_sampler](cfg_dict) # EDITED

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
        
        self.max_iter_per_step = int(self.cfg.max_nfe / self.cfg.max_steps)

    def train_single_step(self, step, sample_dict):
        with torch.enable_grad():
            t_curr, d_t = sm.time_sampler(step)

            latent_noisy = sm.prior.init_latent(self.cfg.batch_size) if step == 0 else sample_dict['xts']

            if latent_noisy.requires_grad:
                latent_noisy = latent_noisy.detach()
            latent_noisy.requires_grad_(True)

            model_pred = sm.prior.compute_velocity_transform_scheduler(latent_noisy, t_curr)
            tweedie = sm.prior.get_tweedie(latent_noisy, model_pred, t_curr)

            target = list()
            decoded_tweedie_list = list()
            for __i in range(0, len(tweedie), self.cfg.minibatch_size):
                cur_batch_size = min(self.cfg.minibatch_size, len(tweedie) - __i)
                decoded_tweedies = sm.prior.decode_latent(tweedie[__i:__i+cur_batch_size])
                decoded_tweedie_list.append(decoded_tweedies.clone().detach())
                target += [sm.corrector.reward_model.preprocess(decoded_tweedies)]

            target = torch.cat(target)
            decoded_tweedies_list = torch.cat(decoded_tweedie_list)
            new_rewards = sm.corrector.reward_model(target, step, decoded_tweedies_list) * 200.0
            derivative = torch.autograd.grad(new_rewards, latent_noisy, torch.ones_like(new_rewards))[0]

            prev_latent_noisy = sm.prior.step(latent_noisy, 
                                        t_curr=t_curr.to(latent_noisy), 
                                        d_t=d_t.to(latent_noisy) / 1000.0,
                                        model_pred=model_pred, # B*N, 4, H, W (v_t)
                                        prev_timestep=None)  
            
            # >>> DAS
            tempering_factor = torch.tensor([min((1 + self.cfg.tempering_gamma) ** (step + 1) - 1, 1.)], device=self.cfg.device, dtype=torch.bfloat16)
            derivative = derivative * tempering_factor

            t_curr_coeff = sm.prior.new_scheduler(t=(t_curr/1000.0))
            alpha_t = t_curr_coeff.alpha_t.item()
            # <<< DAS

            prev_latent_noisy = prev_latent_noisy + self.cfg.guidance_strength * derivative * alpha_t
            
            sample_dict['xts'] = prev_latent_noisy
            # sample_dict['x0s'] = tweedie
            # sample_dict['rewards'] = new_rewards[0].item()
            return sample_dict

    def train(self):
        sample_dict = dict()
        for step in tqdm(range(self.cfg.max_steps), desc="Denoising Step", total=self.cfg.max_steps, leave=False, dynamic_ncols=True):
            sample_dict = self.train_single_step(step, sample_dict)

        # output_filename = os.path.join(self.cfg.root_dir, f"{self.cfg.img_idx:05d}.png")
        output_filename = os.path.join(self.cfg.root_dir, f"{self.cfg.prompt}_{self.cfg.seed:05d}.png")

        final_latent = sample_dict['xts']
        # final_latent = sm.model.image

        final_img = sm.prior.decode_latent(final_latent)
        final_img = final_img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        final_img = (final_img*255).astype('uint8')
        Image.fromarray(final_img).save(output_filename)

        return sample_dict['rewards']