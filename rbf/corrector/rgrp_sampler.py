import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import randint
import random

import torch
import numpy as np 
import torch.nn.functional as F

from rbf.utils.image_utils import torch_to_pil
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_warning, print_error, print_info
from rbf import shared_modules as sm
from rbf.corrector.base import Corrector
from rbf.corrector.reward_model import AestheticRewardModel, CompressionRewardModel, InpaintingRewardModel
from rbf.corrector.reward_model.counting import CountingRewardModel
from rbf.corrector.reward_model.vlm import VLMRewardModel
from rbf.corrector.reward_model.human import HumanRewardModel
from rbf.corrector.reward_model.pickscore import PickScoreRewardModel
from rbf.corrector.reward_model.vqa import VQARewardModel
from rbf.corrector.reward_model.imagereward import ImageRewardRewardModel
from rbf.corrector.reward_model.stylereward import StyleRewardModel


REWARD_FILTERING_METHODs = ["sop", "svdd", "bon", "code", "rbf", "rbf_dps"]

class RGRPSampler(Corrector):
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

        # Differentiable reward config 
        guidance_method: str = "dps"
        gt_image_path: str = ""

        init_n_particles: int = 1


    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

        # BN x D // T
        self.curr_rewards = np.zeros((self.cfg.batch_size * self.cfg.n_particles,))
        self.past_rewards = np.zeros((self.cfg.batch_size,))

        self.curr_potentials = np.zeros((self.cfg.batch_size * self.cfg.n_particles,))
        self.past_potentials = np.zeros((self.cfg.batch_size,))
        
        if self.cfg.reward_score == "aesthetic":
            self.reward_model = AestheticRewardModel(cfg, dtype=torch.bfloat16).to(self.cfg.device).to(torch.bfloat16)
            self.reward_model.requires_grad_(False)

        
        elif self.cfg.reward_score == "compression":
            self.reward_model = CompressionRewardModel(cfg)

        elif self.cfg.reward_score == "inpainting":
            self.reward_model = InpaintingRewardModel(cfg)

        elif self.cfg.reward_score == "counting":
            self.reward_model = CountingRewardModel(cfg)

        elif self.cfg.reward_score == "vlm":
            self.reward_model = VLMRewardModel(cfg)

        elif self.cfg.reward_score == "human":
            self.reward_model = HumanRewardModel(cfg)
        
        elif self.cfg.reward_score == "pickscore":
            self.reward_model = PickScoreRewardModel(cfg)
        
        elif self.cfg.reward_score == "vqa":
            self.reward_model = VQARewardModel(cfg)

        elif self.cfg.reward_score == "imagereward":
            self.reward_model = ImageRewardRewardModel(cfg)
        
        elif self.cfg.reward_score == "style":
            self.reward_model = StyleRewardModel(cfg)

        else:
            raise NotImplementedError(f"Unknown reward score: {self.cfg.reward_score}")
        
        

    def get_potential(
            self, 
            target, 
            particle_idx, 
            step, 
            cur_reward=None,
        ):

        batch_idx = particle_idx // self.cfg.n_particles

        assert self.curr_rewards[particle_idx] == 0, f"{self.curr_rewards[particle_idx]} != 0"
        assert self.curr_potentials[particle_idx] == 0, f"{self.curr_potentials[particle_idx]} != 0"

        # assert self.logging_past_rewards[batch_idx][step] == 0, f"{self.logging_past_rewards[batch_idx][step]} != 0"
        # assert self.logging_past_potentials[batch_idx][step] == 0, f"{self.logging_past_potentials[batch_idx][step]} != 0"

        # Reward
        if cur_reward is None:
            cur_reward = self.reward_model(target, step).item()

        # Potential 
        if self.cfg.filtering_method == "smc":
            if step  == 0:
                cur_potential = np.exp(cur_reward / self.cfg.reward_weight) 
            else:
                prev_reward = self.past_rewards[batch_idx] # [BN]
                cur_potential = np.exp((cur_reward - prev_reward) / self.cfg.reward_weight)

        elif self.cfg.filtering_method.lower() in REWARD_FILTERING_METHODs:
            cur_potential = np.exp(cur_reward / self.cfg.reward_weight)

        else:
            raise NotImplementedError(f"Unknown potential function: {self.cfg.filtering_method}")

        self.curr_rewards[particle_idx] = cur_reward # BN
        self.curr_potentials[particle_idx] = cur_potential

        return cur_potential
    
    def potential(self, tweedie, step):
        p = list()
        assert tweedie.shape[0] == self.cfg.batch_size * self.cfg.n_particles, f"{tweedie.shape[0]} != {self.cfg.batch_size * self.cfg.n_particles}"

        if self.cfg.reward_score == "human":
            # Processed in batch 
            for _i in range(self.cfg.batch_size):
                x0_image = list()
                for _j in range(self.cfg.n_particles):
                    x0_image_i = sm.prior.decode_latent(tweedie[_i * self.cfg.n_particles + _j:_i * self.cfg.n_particles + _j + 1])
                    x0_image.append(torch_to_pil(x0_image_i))

                reward = self.reward_model(x0_image, step)
                for _j, _r in enumerate(reward):
                    potential = self.get_potential(None, _i * self.cfg.n_particles + _j, step, cur_reward=_r)
                    p.append(potential)
        
        elif self.cfg.reward_score in ["vqa", "vqa_old"]:
            target = []
            for _i in range(0, len(tweedie), self.cfg.minibatch_size):
                cur_batch_size = min(self.cfg.minibatch_size, len(tweedie) - _i)
                target += self.reward_model.preprocess(
                    sm.prior.decode_latent(tweedie[_i:_i+cur_batch_size])
                )
            
            assert len(target) == tweedie.shape[0], f"{len(target)} != {tweedie.shape[0]}"

            # x0_image = torch.cat(x0_image, dim=0)
            # assert x0_image.shape[0] == tweedie.shape[0], f"{x0_image.shape[0]} != {tweedie.shape[0]}"
            # target = self.reward_model.preprocess(x0_image)  # Must be List of PIL images 

            rewards = self.reward_model(target, step).cpu()
            assert rewards.shape[0] == tweedie.shape[0], f"{rewards.shape[0]} != {tweedie.shape[0]}"

            for _j, _r in enumerate(rewards):
                potential = self.get_potential(None, _j, step, cur_reward=_r)
                p.append(potential)

        else:
            # Processed individually
            for _i in range(self.cfg.batch_size * self.cfg.n_particles):
                x_0 = tweedie[_i:_i+1]
                x0_image = sm.prior.decode_latent(x_0)

                target = self.reward_model.preprocess(x0_image)
                potential = self.get_potential(target, _i, step)
                p.append(potential)

        assert len(p) == self.cfg.batch_size * self.cfg.n_particles, f"{len(p)} != {self.cfg.batch_size * self.cfg.n_particles}"
        print_info(f"Current potential: {step} ", p) if not sm.OFF_LOG else None
        return p

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
                # for particle_idx in range(self.cfg.batch_size):
                #     x_0 = tweedie[particle_idx:particle_idx+1]
                #     x0_image = sm.prior.decode_latent(x_0)

                #     target = self.reward_model.preprocess(x0_image)

                #     self.get_potential(
                #         target, 
                #         particle_idx, 
                #         step, 
                #         cur_reward=None
                #     )
                p = self.potential(tweedie, step)
                self.past_rewards = self.curr_rewards.copy() # [B]
                self.past_potentials = self.curr_potentials.copy() # [B]

                # self.logging_past_rewards[:, step] = self.curr_rewards # [B, T]
                # self.logging_past_potentials[:, step] = self.curr_potentials # [B, T]

                # Reset the current rewards
                self.curr_rewards = np.zeros_like(self.curr_rewards)
                self.curr_potentials = np.zeros_like(self.curr_potentials)
                
        
            p = self.past_potentials # [B, -1]
            norm_p = p / np.sum(p)
            ess = 1 / (np.sum(norm_p ** 2)) # 1

            print_info(f"Effective sample size: {ess}") if not sm.OFF_LOG else None
            if ess < self.cfg.ess_threshold * self.cfg.batch_size:
                zeta = random.choices(
                    range(self.cfg.batch_size * self.cfg.n_particles), 
                    weights=norm_p, 
                    k=self.cfg.batch_size * self.cfg.n_particles,
                ) # BN

                resample_noisy_sample = noisy_sample[zeta] # BN x D
                # resample_tweedie = tweedie[zeta] # BN x D
                resample_model_pred = model_pred[zeta] # BN x D
                
                self.past_rewards = self.past_rewards[zeta].reshape(self.cfg.batch_size)
                # self.logging_past_rewards = self.logging_past_rewards[zeta]

                self.past_potentials = np.ones_like(self.past_potentials)

                # self.logging_past_potentials[:, step] = 1
                # self.logging_past_potentials = self.logging_past_potentials[zeta]

            else:
                resample_noisy_sample = noisy_sample.clone() # BN x D
                # resample_tweedie = tweedie.clone() # BN x D
                resample_model_pred = model_pred.clone() # BN x D

            # return resample_noisy_sample, resample_tweedie, resample_model_pred
            return resample_noisy_sample, resample_model_pred
    
        elif self.cfg.filtering_method in REWARD_FILTERING_METHODs:
            if self.cfg.filtering_method in ["rbf", "rbf_dps"]:
                # assert step == 0, "Ours only supports step=0 for pre_correct"
                self.cfg.n_particles = self.cfg.init_n_particles
                self.curr_rewards = np.zeros((self.cfg.batch_size * self.cfg.init_n_particles,))
                self.curr_potentials = np.zeros((self.cfg.batch_size * self.cfg.init_n_particles,))

                p = self.potential(tweedie, step)
                p = torch.tensor(p) # BN

                p = p.reshape(self.cfg.batch_size, self.cfg.init_n_particles) # B x N 
                idx = torch.argmax(p, dim=1) # B
                t_arg = torch.arange(self.cfg.batch_size) * self.cfg.init_n_particles + idx # B [0+i1, 1*C+i2, 2*C+i3, 3*C+i4]
                
                resample_noisy_sample = noisy_sample[t_arg] # B x D
                resample_tweedie = tweedie[t_arg] # B x D
                resample_model_pred = model_pred[t_arg] # B x D

                new_init_rewards = torch.tensor(self.curr_rewards[t_arg].reshape(self.cfg.batch_size).copy(), dtype = torch.float32)

                self.cfg.n_particles = 1
                self.curr_rewards = np.zeros((self.cfg.batch_size * self.cfg.n_particles,))
                self.curr_potentials = np.zeros((self.cfg.batch_size * self.cfg.n_particles,))
                return resample_noisy_sample, resample_model_pred, resample_tweedie, new_init_rewards
            
            else:
                return noisy_sample, model_pred

        else:
            raise NotImplementedError(f"Unknown filtering method: {self.cfg.filtering_method}")



    def post_correct(
        self, 
        noisy_sample, # BN x D
        tweedie, # BN x D
        model_pred, # BN x D
        step,
    ):
        
        if self.cfg.filtering_method == "smc":
            assert self.cfg.block_size == 1, "SMC only supports block_size=1"
            assert self.cfg.n_particles == 1, "SMC only supports n_particles=1"

            # for _i in range(self.cfg.batch_size):
            #     x_0 = tweedie[_i:_i+1]
            #     x0_image = sm.prior.decode_latent(x_0)

            #     target = self.reward_model.preprocess(x0_image)
            #     self.get_potential(
            #         target, _i, step
            #     )
            p = self.potential(tweedie, step)
            self.past_rewards = self.curr_rewards.copy()
            self.past_potentials *= self.curr_potentials

            self.curr_rewards = np.zeros_like(self.curr_rewards)
            self.curr_potentials = np.zeros_like(self.curr_potentials)

            return (noisy_sample, tweedie, model_pred)
            

        else:
            if self.cfg.filtering_method == "bon":
                return (noisy_sample, tweedie, model_pred)
            
            else:
                return self.post_correct_rgrp(noisy_sample, tweedie, model_pred, step)
            


    def post_correct_rgrp(
        self, 
        noisy_sample, # BN x D
        tweedie, # BN x D
        model_pred, # BN x D
        step,
    ):
        p = self.potential(tweedie, step)
        p = torch.tensor(p) # BN
        
        # if self.cfg.logging_argmax_index:
        #     print_info("Logging Index: past: {}, curr: {}".format(self.past_rewards[0], self.curr_rewards[:self.cfg.n_particles])) if not sm.OFF_LOG else None
        #     higher = torch.from_numpy(self.past_rewards[0] < self.curr_rewards[:self.cfg.n_particles]);

        #     print_info("Logging Higher: {}".format(higher)) if not sm.OFF_LOG else None
        #     self.higher_index = torch.argmax(higher.int()).item() if torch.any(higher).item() else -1;
        #     self.argmax_index = np.argmax(self.curr_rewards[:self.cfg.n_particles]);
        
        p = p.reshape(self.cfg.batch_size, self.cfg.n_particles) # B x N 
        idx = torch.argmax(p, dim=1) # B
        t_arg = torch.arange(self.cfg.batch_size) * self.cfg.n_particles + idx # B [0+i1, 1*C+i2, 2*C+i3, 3*C+i4]
        
        resample_noisy_sample = noisy_sample[t_arg] # B x D
        resample_tweedie = tweedie[t_arg] # B x D
        resample_model_pred = model_pred[t_arg] # B x D

        assert self.curr_rewards[t_arg].reshape(self.cfg.batch_size).shape == self.past_rewards.shape, f"{self.curr_rewards[t_arg].reshape(self.cfg.batch_size).shape} != {self.past_rewards.shape}"
        # assert self.logging_past_rewards[:, step].shape == self.curr_rewards[t_arg].reshape(self.cfg.batch_size).shape, f"{self.logging_past_rewards[:, step].shape} != {self.curr_rewards[t_arg].reshape(self.cfg.batch_size).shape}"

        self.past_rewards = self.curr_rewards[t_arg].reshape(self.cfg.batch_size) # [B]
        # self.logging_past_rewards[:, step] = self.curr_rewards[t_arg].reshape(self.cfg.batch_size) # [BN, T]
        
        assert self.curr_potentials[t_arg].reshape(self.cfg.batch_size).shape == self.past_potentials.shape, f"{self.curr_potentials[t_arg].reshape(self.cfg.batch_size).shape} != {self.past_potentials.shape}"
        # assert self.logging_past_potentials[:, step].shape == self.curr_potentials[t_arg].reshape(self.cfg.batch_size).shape, f"{self.logging_past_potentials[:, step].shape} != {self.curr_potentials[t_arg].reshape(self.cfg.batch_size).shape}"

        self.past_potentials = self.curr_potentials[t_arg].reshape(self.cfg.batch_size) # [B]
        # self.logging_past_potentials[:, step] = self.curr_potentials[t_arg].reshape(self.cfg.batch_size)

        # Reset the current rewards
        self.curr_rewards = np.zeros_like(self.curr_rewards)
        self.curr_potentials = np.zeros_like(self.curr_potentials)

        return (resample_noisy_sample, resample_tweedie, resample_model_pred) # B x D
    
    
    def final_correct(
        self, 
        tweedie, 
        step,
    ):
        
        assert tweedie.shape[0] == self.cfg.batch_size, f"{tweedie.shape[0]} != {self.cfg.batch_size}"
        

        if self.cfg.filtering_method == "bon":
            assert self.cfg.n_particles == 1, "BON only supports n_particles=1"

            # for _i in range(self.cfg.batch_size * self.cfg.n_particles):
            #     x_0 = tweedie[_i:_i+1]
            #     x0_image = sm.prior.decode_latent(x_0)

            #     target = self.reward_model.preprocess(x0_image)
            #     self.get_potential(target, _i, step)
            p = self.potential(tweedie, step)
            assert self.curr_potentials.shape[0] == self.cfg.batch_size, f"{self.curr_potentials.shape[0]} != {self.cfg.batch_size}"
            p = torch.tensor(self.curr_potentials)

            self.curr_rewards = np.zeros_like(self.curr_rewards)
            self.curr_potentials = np.zeros_like(self.curr_potentials)

        # elif self.cfg.filtering_method == "ours":
        #     p = self.potential(tweedie, step)
        #     p = torch.tensor(self.curr_rewards)

        #     self.curr_rewards = np.zeros_like(self.curr_rewards)
        #     self.curr_potentials = np.zeros_like(self.curr_potentials)

        else:
            assert self.past_potentials.shape[0] == self.cfg.batch_size, f"{self.past_potentials.shape[0]} != {self.cfg.batch_size}"
            p = torch.tensor(self.past_potentials)

            self.curr_rewards = np.zeros_like(self.curr_rewards)
            self.curr_potentials = np.zeros_like(self.curr_potentials)

        max_i = torch.argmax(p).item()

        print("Max potential: ", p[max_i].item()) if not sm.OFF_LOG else None
        # torch_to_pil(sm.prior.decode_latent(tweedie[0:1]))

        tweedie = tweedie[max_i:max_i+1]
    
        return tweedie
    
