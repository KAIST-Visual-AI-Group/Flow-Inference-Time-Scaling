import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

######################################### STABLE DIFFUSION 2.1 #########################################
import numpy as np
import torch
from dataclasses import dataclass

from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler

from rbf.utils.extra_utils import ignore_kwargs
from rbf import shared_modules as sm 


class SD2Prior:
    @ignore_kwargs
    @dataclass
    class Config:
        text_prompt: str 
        negative_prompt: str = None
        batch_size: int = 1
        model_name: str = "stabilityai/stable-diffusion-2-1"
        width: int = 768
        height: int = 768
        guidance_scale: int = 7.5
        root_dir: str = "./results/default"
        max_steps: int = 50
        sample_method: str = "ode"
        diffusion_coefficient: str = "sigma"
        diffusion_norm: float = 1.0
        convert_scheduler: str = None

        disable_debug: bool = False
        log_interval: int = 50
        save_vram: bool = False

        eta: float = 1.0
        minibatch_size: int = 20


    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device("cuda")
        self.cfg = self.Config(**cfg)
        self.cfg.model_name = "stabilityai/stable-diffusion-2-1"
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.cfg.model_name, torch_dtype=torch.float16)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        # self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.device)

        # set timestep
        self.pipeline.scheduler.set_timesteps(self.cfg.max_steps, device=self.device)
        sm.time_sampler.timesteps = self.pipeline.scheduler.timesteps

        # Juse for matching the convention for the mother code, it has nothing to do with the actual behavior of the code
        self.pipeline.scheduler.dt = torch.ones(self.cfg.max_steps, device=self.device) * 0.1 

        self.do_classifier_free_guidance = True if self.cfg.guidance_scale > 1 else False

        self.extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(None, self.cfg.eta)

    def prepare_cond(self, text_prompt=None, negative_prompt=None, _pass=False):
        if not _pass:
            if hasattr(self, "cond"):
                return self.cond
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            text_prompt, self.device, num_images_per_prompt=1, 
            do_classifier_free_guidance=self.do_classifier_free_guidance, negative_prompt=negative_prompt 
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        self.cond = {"prompt_embeds": prompt_embeds}

    def init_latent(self, batch_size):
        num_ch = self.pipeline.unet.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size, num_ch, self.cfg.height, self.cfg.width, torch.float16, device=self.device, generator=None
        )
        return latents

    def predict(self, x_t, timestep, return_dict=False, text_prompt=None, negative_prompt=None):
        self.prepare_cond(text_prompt, negative_prompt)

        noise_pred_list = list()
        for _i in range(0, len(x_t), self.cfg.minibatch_size):
            cur_batch_size = min(self.cfg.minibatch_size, len(x_t) - _i)
            cur_x_t_batch = x_t[_i:_i+cur_batch_size]
            latent_model_input = torch.cat([cur_x_t_batch] * 2) if self.do_classifier_free_guidance else cur_x_t_batch
            # latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t) Not used in DDIM but not sure for other schedulers

            encoder_hidden_state_batch = self.cond["prompt_embeds"].repeat(cur_batch_size, *([1] * (self.cond["prompt_embeds"].dim() - 1)))

            noise_pred = self.pipeline.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_state_batch,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred_list.append(noise_pred)
        noise_pred = torch.cat(noise_pred_list, dim=0)
        return noise_pred

    def decode_latent(self, latent, convert_to_float=True):
        if latent.dim() == 5:
            latent = latent.squeeze(0)    
        image = self.pipeline.vae.decode(latent / self.pipeline.vae.config.scaling_factor, return_dict=False, generator=None)[0]
        image = (image * 0.5 + 0.5).clamp(0, 1)
        if convert_to_float:
            image = image.float()
        return image

    def decode_latent_no_normalize(self, latent):
        image = self.pipeline.vae.decode(latent / self.pipeline.vae.config.scaling_factor, return_dict=False, generator=None)[0]
        return image
    
    def encode_image(self, img_tensor):
        x = (2 * img_tensor - 1).to(self.pipeline.vae.dtype)
        latent = self.pipeline.vae.encode(x, return_dict=False)[0].mode()
        latent = latent * self.pipeline.vae.config.scaling_factor
        return latent
    
    def encode_image_no_normalize(self, img_tensor):
        x = img_tensor.to(self.pipeline.vae.dtype)
        latent = self.pipeline.vae.encode(x, return_dict=False)[0].mode()
        latent = latent * self.pipeline.vae.config.scaling_factor
        return latent
    
    def get_tweedie(self, noisy_sample, model_pred, t):
        self.t_dtype = t.dtype
        output = self.pipeline.scheduler.step(model_pred, t, noisy_sample, **self.extra_step_kwargs, return_dict=True)
        # prev_latent = output.prev_sample
        tweedie = output.pred_original_sample
        # self.prev_latent = prev_latent
        return tweedie

    # def step(self, x, step, model_pred, tweedie, prev_timestep, **kwargs):
    #     return self.prev_latent

    def step(self, x, t_curr, d_t, model_pred, prev_timestep):
        t_curr_ = t_curr.to(self.t_dtype).cpu()
        output = self.pipeline.scheduler.step(model_pred, t_curr_, x, **self.extra_step_kwargs, return_dict=True)
        prev_latent = output.prev_sample
        return prev_latent

    @property
    def dtype(self):
        return self.pipeline.vae.dtype

    def compute_velocity_transform_scheduler(self, latent_noisy, t_curr):
        return self.predict(latent_noisy, t_curr)