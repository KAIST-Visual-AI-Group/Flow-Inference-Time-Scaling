from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import randint

import torch
import torch.nn.functional as F

from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_warning, print_error, print_info
from rbf import shared_modules as sm
from rbf.corrector.base import Corrector
from rbf.corrector.rgrp_sampler import RGRPSampler


class DiffRewardCorrector(RGRPSampler):
    @ignore_kwargs
    @dataclass
    class Config(RGRPSampler.Config):
        strength: float = 1.0
        device: int = 0
        batch_size: int = 1
        n_particles: int = 1

        reward_score: str = 'style'
        guidance_method: str = 'dps'

        disable_debug: bool = False
        log_interval: int = 5


    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)


    def adjust_sample_size(self, t_curr, step):
        return self.cfg.n_particles

    
    def apply_guidance(
        self, 
        noisy_sample, 
        tweedie, 
        step,
    ):
        
        if self.cfg.guidance_method == "dps":
            weight = self.reward_model(tweedie, step)

            # NOTE: Weight is computed as -loss. 
            # Applying guidance with +gradient

            grad = torch.autograd.grad(weight.sum(), noisy_sample)[0]

            prev_latent_noisy = prev_latent_noisy + (self.cfg.strength * grad) / torch.abs(weight.view(-1, * ([1] * (len(prev_latent_noisy.shape) - 1)) ))
        
        else:
            raise NotImplementedError(f"Guidance {self.cfg.guidance} not implemented")

        return prev_latent_noisy


    def post_correct(
        self, 
        noisy_sample,
        tweedie, 
        model_pred,
        step,
    ):

        rgb_tweedie = sm.prior.decode_latent(
            tweedie, convert_to_float=False
        )

        # Apply guidance (DPS/FreeDoM)
        prev_latent_noisy = self.apply_guidance(
            noisy_sample, 
            rgb_tweedie,
            step,
        )

        (
            resample_noisy_sample, 
            resample_tweedie, 
            resample_model_pred
        ) = super().post_correct(
            prev_latent_noisy, 
            tweedie, 
            model_pred,
        )

        return (resample_noisy_sample, resample_tweedie, resample_model_pred) # B x D