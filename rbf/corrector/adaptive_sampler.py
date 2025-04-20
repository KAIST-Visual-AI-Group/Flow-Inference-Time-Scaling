import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import randint
import random
import math 
import torch
import numpy as np 
import torch.nn.functional as F

from rbf.utils.image_utils import torch_to_pil
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_warning, print_error, print_info
from rbf import shared_modules as sm
from rbf.corrector.base import Corrector
from rbf.corrector.rgrp_sampler import RGRPSampler


REWARD_FILTERING_METHODs = ["sop", "svdd", "bon", "ours", "code"]

class AdaptiveSampler(RGRPSampler):
    @ignore_kwargs
    @dataclass
    class Config(Corrector.Config):
        batch_size: int = 20
        device: int = 0
        minibatch_size: int = 5
        n_particles: int = 20 # Particle size
        reward_score: str = 'compression'
        
        ckpt_root: str = ""
        ckpt: str = "sac+logos+ava1-l14-linearMSE.pth"

        reward_weight: float = 1.0
        max_steps: int = 50
        ess_threshold: float = 1.0
        
        disable_debug: bool = False
        log_interval: int = 5

        class_names: str = ""
        class_gt_counts: str = "" 
        count_reward_model: str = ""

        block_size: int = 1
        filtering_method: str = None

        # Adaptive sampler config
        max_nfe: int = 500
        adjust_sample_method: str = "linear"

    

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

        print("Loaded adaptive sampler") if not sm.OFF_LOG else None

        if self.cfg.adjust_sample_method == "linear":
            nfe = self.cfg.max_nfe
            N = self.cfg.max_steps
            com_diff = int((nfe - N) * 2 / (N * (N - 1)))

            init_particles = int((nfe + com_diff * N * (N - 1) / 2) / N)
            init_sched = [init_particles - i*com_diff for i in range(N)]

            left_over = nfe - sum(init_sched)
            for i in range(left_over):
                init_sched[i] += 1

            assert sum(init_sched) == nfe, f"Sum of initial schedule {sum(init_sched)} is not equal to max nfe {nfe}"

            if self.cfg.filtering_method == "smc":
                init_sched.append(init_sched[-1])

            self.adaptive_particle_shcedule = init_sched                

            # print("="*100)
            # print("Adaptive particle schedule: ", self.adaptive_particle_shcedule)
            # print("="*100)

        else:
            raise NotImplementedError(f"Unknown adjust sample method: {self.cfg.adjust_sample_method}")
    
    def adjust_sample_size(
        self,
        t_curr,
        step, 
    ):
        # TODO: return adaptive n particle 
        # adjust rewards / potentials

        # t = t_curr[0].item()
        # if step == self.cfg.max_steps - 1:
        #     return self.cfg.n_particles

        # if self.cfg.adjust_sample_method == "linear":
        #     new_n_particles = math.ceil((t / 1000.0) * self.cfg.n_particles)
        # elif self.cfg.adjust_sample_method == "square":
        #     new_n_particles = math.ceil((1.0 - (1.0 - t / 1000.0) ** 2) ** 0.5 * self.cfg.n_particles)
        # elif self.cfg.adjust_sample_method == "debug":
        #     new_n_particles = (self.cfg.n_particles - 1)
        # elif self.cfg.adjust_sample_method == "identity":
        #     new_n_particles = self.cfg.n_particles
        # else:
        #     raise NotImplementedError(f"Unknown adjust sample method: {self.cfg.adjust_sample_method}") 
    
        # new_n_particles = max(1, new_n_particles)
        
        # print(f"Number of particles adjusted at {t} from {self.cfg.n_particles} to {new_n_particles}") if not sm.OFF_LOG else None
        
        new_n_particles = self.adaptive_particle_shcedule[step]
        
        if self.cfg.filtering_method == "svdd":
            self.cfg.n_particles = new_n_particles
            self.reward_model.cfg.n_particles = new_n_particles
            self.curr_rewards = np.zeros((self.cfg.batch_size * self.cfg.n_particles,))
            self.past_rewards = np.zeros((self.cfg.batch_size,))

            self.curr_potentials = np.zeros((self.cfg.batch_size * self.cfg.n_particles,))
            self.past_potentials = np.zeros((self.cfg.batch_size,))
        elif self.cfg.filtering_method == "smc":
            self.smc_change_cur_batch_size(new_n_particles)
            self.curr_rewards = np.zeros((self.cfg.batch_size * self.cfg.n_particles,))
            self.curr_potentials = np.zeros((self.cfg.batch_size * self.cfg.n_particles,))

        else:
            raise NotImplementedError(f"Unknown filtering method: {self.cfg.filtering_method}")
        
        return new_n_particles

    def smc_change_cur_batch_size(self, new_batch_size):
        self.reward_model.cfg.batch_size = new_batch_size
        self.cfg.batch_size = new_batch_size
        sm.time_sampler.cfg.batch_size = new_batch_size

    def pre_correct(
        self, 
        noisy_sample, # BN x D
        tweedie, # BN x D
        model_pred, # BN x D
        step, # for safer code
    ):

        if self.cfg.filtering_method == "smc":
            assert self.cfg.block_size == 1, "SMC only supports block_size=1"
            assert self.cfg.n_particles == 1, "SMC only supports n_particles=1"

            if step == 0:
                p = self.potential(tweedie, step)
                self.past_rewards = self.curr_rewards.copy() # [B]
                self.past_potentials = self.curr_potentials.copy() # [B]

                # Reset the current rewards
                self.curr_rewards = np.zeros_like(self.curr_rewards)
                self.curr_potentials = np.zeros_like(self.curr_potentials)
        
            p = self.past_potentials # [B, -1]
            norm_p = p / np.sum(p)
            ess = 1 / (np.sum(norm_p ** 2)) # 1

            print_info(f"Effective sample size: {ess}") if not sm.OFF_LOG else None
            next_batch_size = self.adaptive_particle_shcedule[step + 1]
            
            assert self.cfg.ess_threshold==1.0, "Adaptive SMC only supports ess_threshold=1.0"
            if ess <= self.cfg.ess_threshold * self.cfg.batch_size:
                zeta = random.choices(
                    range(self.cfg.batch_size), 
                    weights=norm_p, 
                    k=next_batch_size,
                ) # BN

                resample_noisy_sample = noisy_sample[zeta] # BN x D
                resample_model_pred = model_pred[zeta] # BN x D
                
                self.past_rewards = self.past_rewards[zeta].reshape(next_batch_size)

                # self.past_potentials = np.ones_like(self.past_potentials)
                self.past_potentials = np.ones(next_batch_size)
                self.curr_rewards = np.zeros(next_batch_size)
                self.curr_potentials = np.zeros(next_batch_size)

            else:
                resample_noisy_sample = noisy_sample.clone() # BN x D
                resample_model_pred = model_pred.clone() # BN x D

            return resample_noisy_sample, resample_model_pred
    
        elif self.cfg.filtering_method in REWARD_FILTERING_METHODs:
            # return noisy_sample, tweedie, model_pred
            return noisy_sample, model_pred

        else:
            raise NotImplementedError(f"Unknown filtering method: {self.cfg.filtering_method}")