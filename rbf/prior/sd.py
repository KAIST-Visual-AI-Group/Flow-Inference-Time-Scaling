from typing import Dict, Literal, Tuple, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)

from rbf.utils.extra_utils import (
    ignore_kwargs,
)
from rbf.utils.print_utils import print_info, print_warning, print_error

import rbf.shared_modules as sm
from rbf.prior.base import Prior, NEGATIVE_PROMPT


class StableDiffusionPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        device: int = 0
        batch_size: int = 1
        model_name: str = "runwayml/stable-diffusion-v1-5"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = ""
        width: int = 512
        height: int = 512
        guidance_scale: int = 7.5
        root_dir: str = "./results/default"
        max_steps: int = 50

        minibatch_size: int = 10
        eta: float = 1.0

        use_dpo: bool = True
        precision: str = "fp16"
        sd_model: str = ""


    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        print("Loading ", self.cfg.model_name)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.model_name, subfolder="scheduler"
        )

        # -------------------------------------------------------------------
        # Precision
        # -------------------------------------------------------------------
        if self.cfg.precision == "fp32":
            _dtype = torch.float32

        elif self.cfg.precision == "fp16":
            _dtype = torch.float16

        else:
            raise NotImplementedError("Only fp32 and fp16 are supported")
        

        # -------------------------------------------------------------------
        # Load the UNet model
        # -------------------------------------------------------------------
        print_info("Loading Stable Diffusion", self.cfg.sd_model, self.cfg.precision)
        if self.cfg.sd_model in "sd15":
            model_id = "runwayml/stable-diffusion-v1-5"
            unet_id = "mhdang/dpo-sd1.5-text2image-v1"
            self.cfg.guidance_scale = 7.5
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                scheduler=self.ddim_scheduler,
                torch_dtype=_dtype,
            ).to(self.cfg.device)

        elif self.cfg.sd_model == "sd2":
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.cfg.model_name,
                scheduler=self.ddim_scheduler,
                torch_dtype=_dtype,
            ).to(self.cfg.device)
            
        elif self.cfg.sd_model == "sdxl":
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            unet_id = "mhdang/dpo-sdxl-text2image-v1"
            self.cfg.guidance_scale = 5.0
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=_dtype
            ).to(self.cfg.device)

        else:
            raise NotImplementedError("Only sd15 and sdxl are supported")
        

        # if self.cfg.use_dpo:
        #     print_info("Loading Diffusion-DPO")
        #     unet = UNet2DConditionModel.from_pretrained(
        #         unet_id, subfolder="unet", torch_dtype=self.pipeline.dtype,
        #     ).to(self.cfg.device)
        #     self.pipeline.unet = unet

        self.ddim_scheduler.set_timesteps(self.cfg.max_steps)
        self.ddim_scheduler.alphas_cumprod = self.ddim_scheduler.alphas_cumprod.to(self.cfg.device).to(self.pipeline.dtype)

        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        self.pipeline.unet.eval()
        self.pipeline.vae.eval()
        self.pipeline.text_encoder.eval()

        self.nfe = 0

        
    @property
    def rgb_res(self):
        return 1, 3, 512, 512
    
    @property
    def latent_res(self):
        return 1, 4, 64, 64

    def prepare_cond(self, text_prompt=None, negative_prompt=None, _pass=False):
        if not _pass:
            if hasattr(self, "cond"):
                return self.cond 
        
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        print_info("Encoding text prompt", text_prompt)

        text_embeddings = self.encode_text(
            text_prompt, negative_prompt=negative_prompt
        )  # neg, pos
        
        neg, pos = text_embeddings.chunk(2)

        self.cond = {
            "neg": neg,
            "pos": pos, 
        }

        return self.cond

    
    def init_latent(self, batch_size, latents=None):
        num_channels_latents = self.pipeline.unet.config.in_channels

        latents = self.pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            self.cfg.height,
            self.cfg.width,
            self.pipeline.dtype,
            self.pipeline.device,
            generator = None,
            latents = latents,
        )

        return latents
    



    def predict(
        self, 
        x_t, 
        timestep, 
        guidance_scale=None, 
        return_dict=False, 
        text_prompt=None, 
        negative_prompt=None,
    ):

        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        self.prepare_cond(text_prompt, negative_prompt)
        
        noise_pred = []
        for _i in range(0, len(x_t), self.cfg.minibatch_size):
            cur_batch_size = min(self.cfg.minibatch_size, len(x_t) - _i)
            cur_x_t_batch = x_t[_i:_i+cur_batch_size]
            cur_t = timestep[_i:_i+cur_batch_size].view(-1)

            cur_cond = {}
            for k, v in self.cond.items():
                # neg, pos embeddings 
                cur_cond[k] = v.repeat(cur_batch_size, *([1] * (v.dim() - 1)))

            cur_cond["encoder_hidden_states"] = torch.cat([cur_cond["neg"], cur_cond["pos"]], dim=0)
            cur_cond.pop('neg', None)
            cur_cond.pop('pos', None)

            cfg_cur_x_t_batch = torch.cat([cur_x_t_batch] * 2)
            cfg_cur_t = torch.cat([cur_t] * 2)

            cur_noise_pred = self.pipeline.unet(
                cfg_cur_x_t_batch, cfg_cur_t,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
                **cur_cond,
            )[0]

            cur_noise_pred_uncond, cur_noise_pred_text = cur_noise_pred.chunk(2)
            cur_noise_pred = cur_noise_pred_uncond + self.cfg.guidance_scale * (
                cur_noise_pred_text - cur_noise_pred_uncond
            )

            self.nfe += len(cfg_cur_x_t_batch)
            noise_pred.append(cur_noise_pred)

        noise_pred = torch.cat(noise_pred, dim=0)

        if return_dict:
            return {
                "noise_pred": noise_pred,
            }
        return noise_pred
    


    def get_variance(self, alpha_prod_t, alpha_prod_t_prev):
        # alpha_prod_t = self.alphas_cumprod[timestep]
        # alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance


    def step(
        self, 
        x, 
        t_curr,
        d_t,
        model_pred=None, 
        prev_timestep=None,
    ):
        
        squeeze_t_curr = t_curr.long().view(-1)
        squeeze_prev_timestep = prev_timestep.long().view(-1)

        _alphas = []
        _alphas_prev = []
        for _i in range(len(squeeze_t_curr)):
            _t = squeeze_t_curr[_i]
            _pt = squeeze_prev_timestep[_i]

            cur_alpha = self.ddim_scheduler.alphas_cumprod[_t] if _t >= 0 else self.ddim_scheduler.final_alpha_cumprod
            _alphas.append(cur_alpha)

            cur_alpha_prev = self.ddim_scheduler.alphas_cumprod[_pt] if _pt >= 0 else self.ddim_scheduler.final_alpha_cumprod
            _alphas_prev.append(cur_alpha_prev)

        alpha_prod_t = torch.stack(_alphas).view(t_curr.shape)
        alpha_prod_t_prev = torch.stack(_alphas_prev).view(prev_timestep.shape)

        pred_original_sample = self.get_tweedie(x, model_pred, t_curr)
        
        variance = self.get_variance(
            alpha_prod_t, alpha_prod_t_prev
        )
        std_dev_t = self.cfg.eta * variance ** (0.5)
        
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_pred
        prev_latent = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        variance_noise = torch.randn_like(prev_latent)
        variance = std_dev_t * variance_noise

        return prev_latent + variance
    

    def compute_velocity_transform_scheduler(self, x, t, **extras):
        u_r = self.predict(x, (t).to(x.dtype))

        return u_r
    

    # def get_tweedie(self, noisy_sample, model_pred, t):
    #     timestep = (t).to(model_pred)

    #     r_scheduler_output = cur_scheduler(t=timestep.float())

    #     alpha_r = r_scheduler_output.alpha_t.to(model_pred.dtype)
    #     sigma_r = r_scheduler_output.sigma_t.to(model_pred.dtype)
    #     d_alpha_r = r_scheduler_output.d_alpha_t.to(model_pred.dtype)
    #     d_sigma_r = r_scheduler_output.d_sigma_t.to(model_pred.dtype)

    #     numer = (sigma_r * model_pred) - (d_sigma_r * noisy_sample)
    #     denom = (d_alpha_r * sigma_r) - (d_sigma_r * alpha_r)

    #     return numer / denom


    
    def tau_func(
        self, 
        t_curr, 
        d_t,
    ):
        tau = (t_curr / 1000.0) * (d_t * self.cfg.tau_norm)
        # assert tau >= 0, f"Invalid tau value {tau}"

        return tau 


    def sample(self, text_prompt=None):
        if text_prompt is None:
            text_prompt = self.cfg.text_prompt

        self.prepare_cond()
        with torch.no_grad():
            images = self.pipeline(
                [text_prompt], negative_prompt=[self.cfg.negative_prompt]
            ).images
        return images
    
    def fast_sample(self, x_t, timesteps, guidance_scale=None, text_prompt=None, negative_prompt=None):
        self.fast_scheduler.set_timesteps(timesteps=timesteps)
        for t in timesteps:
            noise_pred = self.predict(x_t, t, guidance_scale, text_prompt, negative_prompt)
            x_t = self.fast_scheduler.step(noise_pred, t, x_t, return_dict=False)[0]

        return x_t