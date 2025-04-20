# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import inspect
import numpy as np
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from rbf.prior.flux import FlowMatchEulerDiscreteScheduler
from rbf.utils.extra_utils import ignore_kwargs, weak_lru
from rbf.utils.print_utils import print_info, print_note
from rbf.prior.base import Prior, NEGATIVE_PROMPT
from rbf import shared_modules as sm 
from rbf.utils.image_utils import torch_to_pil_batch, image_grid


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
#     """
#     Euler scheduler.

#     This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
#     methods the library implements for all schedulers such as loading and saving.

#     Args:
#         num_train_timesteps (`int`, defaults to 1000):
#             The number of diffusion steps to train the model.
#         timestep_spacing (`str`, defaults to `"linspace"`):
#             The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
#             Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
#         shift (`float`, defaults to 1.0):
#             The shift value for the timestep schedule.
#     """

#     _compatibles = []
#     order = 1

#     @register_to_config
#     def __init__(
#         self,
#         num_train_timesteps: int = 1000,
#         shift: float = 1.0,
#         use_dynamic_shifting=False,
#         base_shift: Optional[float] = 0.5,
#         max_shift: Optional[float] = 1.15,
#         base_image_seq_len: Optional[int] = 256,
#         max_image_seq_len: Optional[int] = 4096,
#         invert_sigmas: bool = False,
#     ):

#         timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
#         timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

#         sigmas = timesteps / num_train_timesteps
#         if not use_dynamic_shifting:
#             # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
#             sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

#         self.timesteps = sigmas * num_train_timesteps

#         self._step_index = None
#         self._begin_index = None

#         self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
#         self.sigma_min = self.sigmas[-1].item()
#         self.sigma_max = self.sigmas[0].item()

#         # Use for score computation
#         self.d_sigmas = 1 # d\sigma / dt
#         self.d_alphas = -1 # d\alpha / dt


#     @property
#     def step_index(self):
#         """
#         The index counter for current timestep. It will increase 1 after each scheduler step.
#         """
#         return self._step_index

#     @property
#     def begin_index(self):
#         """
#         The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
#         """
#         return self._begin_index

#     # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
#     def set_begin_index(self, begin_index: int = 0):
#         """
#         Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

#         Args:
#             begin_index (`int`):
#                 The begin index for the scheduler.
#         """
#         self._begin_index = begin_index

#     def scale_noise(
#         self,
#         sample: torch.FloatTensor,
#         timestep: Union[float, torch.FloatTensor],
#         noise: Optional[torch.FloatTensor] = None,
#     ) -> torch.FloatTensor:
#         """
#         Forward process in flow-matching

#         Args:
#             sample (`torch.FloatTensor`):
#                 The input sample.
#             timestep (`int`, *optional*):
#                 The current timestep in the diffusion chain.

#         Returns:
#             `torch.FloatTensor`:
#                 A scaled input sample.
#         """
#         # Make sure sigmas and timesteps have the same device and dtype as original_samples
#         sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

#         if sample.device.type == "mps" and torch.is_floating_point(timestep):
#             # mps does not support float64
#             schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
#             timestep = timestep.to(sample.device, dtype=torch.float32)
#         else:
#             schedule_timesteps = self.timesteps.to(sample.device)
#             timestep = timestep.to(sample.device)

#         # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
#         if self.begin_index is None:
#             step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
#         elif self.step_index is not None:
#             # add noise is called after first denoising step (for inpainting)
#             step_indices = [self.step_index] * timestep.shape[0]
#         else:
#             # add noise is called before first denoising step to create initial latent(img2img)
#             step_indices = [self.begin_index] * timestep.shape[0]

#         sigma = sigmas[step_indices].flatten()
#         while len(sigma.shape) < len(sample.shape):
#             sigma = sigma.unsqueeze(-1)

#         sample = sigma * noise + (1.0 - sigma) * sample

#         return sample

#     def _sigma_to_t(self, sigma):
#         return sigma * self.config.num_train_timesteps

#     def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
#         return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

#     def set_timesteps(
#         self,
#         num_inference_steps: int = None,
#         device: Union[str, torch.device] = None,
#         sigmas: Optional[List[float]] = None,
#         mu: Optional[float] = None,
#     ):
#         """
#         Sets the discrete timesteps used for the diffusion chain (to be run before inference).

#         Args:
#             num_inference_steps (`int`):
#                 The number of diffusion steps used when generating samples with a pre-trained model.
#             device (`str` or `torch.device`, *optional*):
#                 The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
#         """

#         if self.config.use_dynamic_shifting and mu is None:
#             raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

#         if sigmas is None:
#             self.num_inference_steps = num_inference_steps
#             timesteps = np.linspace(
#                 self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
#             )

#             sigmas = timesteps / self.config.num_train_timesteps

#         if self.config.use_dynamic_shifting:
#             sigmas = self.time_shift(mu, 1.0, sigmas)
#         else:
#             sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

#         sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
#         timesteps = sigmas * self.config.num_train_timesteps

#         if self.config.invert_sigmas:
#             sigmas = 1.0 - sigmas
#             timesteps = sigmas * self.config.num_train_timesteps
#             sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
#         else:
#             sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

#         self.timesteps = timesteps.to(device=device)
#         self.sigmas = sigmas
#         self._step_index = None
#         self._begin_index = None

#         # dt initialization
#         self.alphas = 1 - self.sigmas
#         self.d_sigma = self.sigmas[:-1] - self.sigmas[1:] # positive step size
#         self.dt = self._sigma_to_t(self.d_sigma)
        

#     def index_for_timestep(self, timestep, schedule_timesteps=None):
#         if schedule_timesteps is None:
#             schedule_timesteps = self.timesteps

#         print(timestep, schedule_timesteps)
#         indices = (schedule_timesteps == timestep).nonzero()

#         # The sigma index that is taken for the **very** first `step`
#         # is always the second index (or the last index if there is only 1)
#         # This way we can ensure we don't accidentally skip a sigma in
#         # case we start in the middle of the denoising schedule (e.g. for image-to-image)
#         pos = 1 if len(indices) > 1 else 0

#         return indices[pos].item()

#     def _init_step_index(self, timestep):
#         if self.begin_index is None:
#             if isinstance(timestep, torch.Tensor):
#                 timestep = timestep.to(self.timesteps.device)
#             self._step_index = self.index_for_timestep(timestep)
#         else:
#             self._step_index = self._begin_index

    
#     def convert_velocity_to_score(
#         self,
#         model_output: torch.FloatTensor, 
#         step: Union[float, torch.FloatTensor],
#         sample: torch.FloatTensor,
#         convert_scheduler=None,
#         new_scheduler=None,
#         original_scheduler=None,
#     ):

#         new_t = self.timesteps[step] / 1000

#         if convert_scheduler is not None:
#             print_note("[***] Using convert scheduler in computing score from velocity")
#             cur_scheduler = new_scheduler

#         else:
#             print("Not using convert scheduler in computing score from velocity")
#             cur_scheduler = original_scheduler

#         new_schedule_coeffs = cur_scheduler(new_t.float())
        
#         sigma_t = new_schedule_coeffs.sigma_t.to(model_output.dtype)
#         d_sigma_t = new_schedule_coeffs.d_sigma_t.to(model_output.dtype)
#         alpha_t = new_schedule_coeffs.alpha_t.to(model_output.dtype)
#         d_alpha_t = new_schedule_coeffs.d_alpha_t.to(model_output.dtype)
        
#         # if convert_scheduler == "vp":
#         #     print_note("[***] Using convert scheduler in computing score from velocity")
#         #     # new_t = self.get_new_timestep(self.timesteps[step], original_scheduler, new_scheduler)
#         #     new_t = self.timesteps[step] / 1000

#         #     new_schedule_coeffs = new_scheduler(new_t)

#         #     sigma_t = new_schedule_coeffs.sigma_t
#         #     d_sigma_t = new_schedule_coeffs.d_sigma_t

#         #     alpha_t = new_schedule_coeffs.alpha_t
#         #     d_alpha_t = new_schedule_coeffs.d_alpha_t
        
#         # else:
#         #     print("Not using convert scheduler in computing score from velocity")
            
#         #     # Flow Model
#         #     sigma_t = self.sigmas[step]
#         #     d_sigma_t = self.d_sigmas # integer

#         #     alpha_t = self.alphas[step]
#         #     d_alpha_t = self.d_alphas # integer

#         reverse_alpha_ratio = alpha_t / d_alpha_t

#         var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
#         score = (reverse_alpha_ratio * model_output - sample) / var

#         return score

#     def get_new_timestep(
#         self,
#         t,
#         original_scheduler,
#         new_scheduler,
#     ):
        
#         r = t / 1000.0
        
#         r_scheduler_output = new_scheduler(t=r)

#         alpha_r = r_scheduler_output.alpha_t
#         sigma_r = r_scheduler_output.sigma_t
#         # print("Current", "alpha_r", alpha_r.item(), "sigma_r", sigma_r.item(), \
#         #       "d_alpha_r", d_alpha_r.item(), "d_sigma_r", d_sigma_r.item())

#         new_t = original_scheduler.snr_inverse(alpha_r / sigma_r)
#         print("Current r", r.item(), "-->", "New t", new_t.item())

#         return new_t
        
    
#     # Diffuse
#     def get_sde_diffuse(
#         self, 
#         step,
#         diffusion_coefficient="sigma", 
#         diffusion_norm=1, 
#         convert_scheduler=None,
#         new_scheduler=None,
#         original_scheduler=None,
#     ):
        
#         if convert_scheduler is not None:
#             print_note("[***] Convert scheduler", convert_scheduler, "Adding noise", diffusion_coefficient, "norm", diffusion_norm)
#             cur_scheduler = new_scheduler

#         else:
#             print("No Convert scheduler", "Adding noise", diffusion_coefficient, "norm", diffusion_norm)
#             cur_scheduler = original_scheduler

#         new_t = self.timesteps[step] / 1000

#         if diffusion_coefficient == "sigma":
#             return diffusion_norm * cur_scheduler(new_t).sigma_t
        
#         elif diffusion_coefficient == "constant":
#             return diffusion_norm
        
#         elif diffusion_coefficient == "sin":
#             return diffusion_norm * torch.sin(torch.pi/2 * new_t)
        
#         elif diffusion_coefficient == "linear":
#             return diffusion_norm * new_t
        
#         elif diffusion_coefficient == "exp":
#             return diffusion_norm * (new_t **2)

#         elif diffusion_coefficient == "clipping":
#             return diffusion_norm * max(0.0, 2.0 * (new_t - 0.5))
        
#         elif diffusion_coefficient == "tan":
#             clamp_new_t = torch.clamp(new_t, min=0, max=0.95)
#             return diffusion_norm * torch.tan(torch.pi/2 * clamp_new_t)
        
#         else:
#             raise NotImplementedError(f"Diffusion coefficient {diffusion_coefficient} not implemented")
        

#         # if convert_scheduler == "vp":
#         #     print_note("[***] Convert scheduler", convert_scheduler, "Adding noise", diffusion_coefficient, "norm", diffusion_norm)
#         #     assert new_scheduler is not None and original_scheduler is not None, f"{new_scheduler}, {original_scheduler}"

#         #     # new_t = self.get_new_timestep(
#         #     #     self.timesteps[step], original_scheduler, new_scheduler
#         #     # )
#         #     new_t = self.timesteps[step] / 1000

#         #     if diffusion_coefficient == "sigma":
#         #         return diffusion_norm * new_scheduler(new_t).sigma_t
            
#         #     elif diffusion_coefficient == "constant":
#         #         return diffusion_norm
            
#         #     elif diffusion_coefficient == "sin":
#         #         return diffusion_norm * torch.sin(torch.pi/2 * new_t)
            
#         #     elif diffusion_coefficient == "linear":
#         #         return diffusion_norm * new_t
            
#         #     elif diffusion_coefficient == "exp":
#         #         return diffusion_norm * (new_t **2)
            
#         #     else:
#         #         raise NotImplementedError(f"Diffusion coefficient {diffusion_coefficient} not implemented")

#         # else:
#         #     print("No Convert scheduler", "Adding noise", diffusion_coefficient, "norm", diffusion_norm)

#         #     if diffusion_coefficient == "sigma":
#         #         return diffusion_norm * self.sigmas[step]
            
#         #     elif diffusion_coefficient == "constant":
#         #         return diffusion_norm
            
#         #     elif diffusion_coefficient == "sin":
#         #         cur_t = self.timesteps[step] / 1000
#         #         return diffusion_norm * torch.sin(torch.pi/2 * cur_t)
            
#         #     elif diffusion_coefficient == "linear":
#         #         return diffusion_norm * self.timesteps[step] / 1000
            
#         #     elif diffusion_coefficient == "exp":
#         #         return diffusion_norm * (self.timesteps[step] / 1000) ** 2

#         #     else:
#         #         raise NotImplementedError(f"Diffusion coefficient {diffusion_coefficient} not implemented")


#     def get_ode_diffuse(self):
#         return 0

#     def get_diffuse(
#         self, 
#         step,
#         sample_method, 
#         diffusion_coefficient="sigma", 
#         diffusion_norm=1, 
#         convert_scheduler=None,
#         new_scheduler=None, 
#         original_scheduler=None,
#     ):
        
#         if sample_method == "ode":
#             return self.get_ode_diffuse()
        
#         elif sample_method == "sde":
#             return self.get_sde_diffuse(
#                 step=step,
#                 diffusion_coefficient=diffusion_coefficient, 
#                 diffusion_norm=diffusion_norm, 
#                 convert_scheduler=convert_scheduler,
#                 new_scheduler=new_scheduler,
#                 original_scheduler=original_scheduler,
#             )

#         else:
#             raise ValueError(f"Unknown sample method {sample_method}")
    
#     # Drift
#     def get_sde_drift(
#         self, 
#         velocity, 
#         step, 
#         sample, 
#         sample_method, 
#         diffusion_coefficient="sigma", 
#         diffusion_norm=1,
#         convert_scheduler=None,
#         new_scheduler=None,
#         original_scheduler=None,
#         diffuse=None, 
#     ):

#         if diffuse is None:
#             diffuse = self.get_diffuse(
#                 step,
#                 sample_method, 
#                 diffusion_coefficient=diffusion_coefficient, 
#                 diffusion_norm=diffusion_norm,
#                 convert_scheduler=convert_scheduler,
#                 new_scheduler=new_scheduler,
#                 original_scheduler=original_scheduler,
#             )

#         score = self.convert_velocity_to_score(
#             velocity, step, sample, 
#             convert_scheduler=convert_scheduler, 
#             new_scheduler=new_scheduler, 
#             original_scheduler=original_scheduler,
#         )

#         # print_info("score", score.mean().item(), "velocity", velocity.mean().item())
#         drift = -velocity + (0.5 * diffuse ** 2) * score

#         return drift
    
#     def get_ode_drift(self, velocity):
#         return -velocity
        
#     def get_drift(
#         self, 
#         noisy_sample, 
#         model_pred, 
#         step, 
#         sample_method, 
#         diffusion_coefficient="sigma", 
#         diffusion_norm=1,
#         convert_scheduler=None,
#         new_scheduler=None,
#         original_scheduler=None,
#         diffuse=None,
#     ):

#         if sample_method == "ode":
#             drift = self.get_ode_drift(model_pred)

#         elif sample_method == "sde":
#             # if self.step_index is None:
#             #     self._init_step_index(timestep)

#             drift = self.get_sde_drift(
#                 model_pred, 
#                 step, 
#                 noisy_sample, 
#                 sample_method, 
#                 diffusion_coefficient=diffusion_coefficient, 
#                 diffusion_norm=diffusion_norm,
#                 convert_scheduler=convert_scheduler,
#                 new_scheduler=new_scheduler,
#                 original_scheduler=original_scheduler,
#                 diffuse=diffuse, 
#             )

#         else:
#             raise NotImplementedError("Invalid sample method")
        
#         return drift
        

#     def __len__(self):
#         return self.config.num_train_timesteps
    


class InstaFlowPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Rectified Flow and Euler discretization.
    This customized pipeline is based on StableDiffusionPipeline from the official Diffusers library (0.21.4)

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        # Editied 2025.02.07
        # latents = latents * self.scheduler.init_noise_sigma

        latents = latents
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps = [(1. - i/num_inference_steps) * 1000. for i in range(num_inference_steps)]

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        dt = 1.0 / num_inference_steps

        # 7. Denoising loop of Euler discretization from t = 0 to t = 1
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 

                v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance 
                if do_classifier_free_guidance:
                    v_pred_neg, v_pred_text = v_pred.chunk(2)
                    v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

                latents = latents + dt * v_pred 

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)



        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class InstaFlowPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        device: int = 0
        batch_size: int = 1
        model_name: str = "XCLIU/2_rectified_flow_from_sd_1_5"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = "a"
        width: int = 512
        height: int = 512
        guidance_scale: int = 1.5
        root_dir: str = "./results/default"
        max_steps: int = 4
        sample_method: str = "ode"
        diffusion_coefficient: str = "sigma"
        diffusion_norm: float = 1.0
        convert_scheduler: str = None

        disable_debug: bool = False
        log_interval: int = 50
        save_vram: bool = False

        scheduler_n: float = 0.5
        minibatch_size: int = 5

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        if self.cfg.sample_method == "sde":
            assert self.cfg.diffusion_coefficient is not None
            assert self.cfg.diffusion_norm is not None

        print("Using prior model: ", self.cfg.model_name)
        self.pipeline = InstaFlowPipeline.from_pretrained(
            self.cfg.model_name, 
            torch_dtype=torch.float16,
            # torch_dtype=torch.float32,
        )

        if not self.cfg.save_vram:
            # Save VRAM performs CPU offloading
            self.pipeline = self.pipeline.to(self.cfg.device)

        self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        print_info(self.cfg.text_prompt)
        print_info(f"Sampling method: {self.cfg.sample_method}")

        sigmas = np.linspace(1.0, 1 / self.cfg.max_steps, self.cfg.max_steps)
        # This can be overriden in time_sampler for custom timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            self.cfg.max_steps,
            self.pipeline.device,
            None,
            sigmas,
            # mu=mu,
        )

        from rbf.prior.denoise_schedulers import CondOTScheduler
        self.original_scheduler = CondOTScheduler(device=self.pipeline.device)
        self.new_scheduler = None

        print(f"[*] Original scheduler set to {self.original_scheduler.__class__.__name__}")
        if self.cfg.convert_scheduler == "vp":
            from rbf.prior.denoise_schedulers import VPScheduler
            self.new_scheduler = VPScheduler(device=self.pipeline.device)

            print_note(f"[***] Using scheduler conversion to {self.new_scheduler.__class__.__name__}")

        
        elif self.cfg.convert_scheduler == "polynomial":
            from rbf.prior.denoise_schedulers import PolynomialConvexScheduler
            self.new_scheduler = PolynomialConvexScheduler(
                n=self.cfg.scheduler_n,
                device=self.pipeline.device,
            )

            print_note(f"[***] Using scheduler conversion to {self.new_scheduler.__class__.__name__} scheduler_n: {self.cfg.scheduler_n}")


        elif self.cfg.convert_scheduler == "general":
            raise NotImplementedError("General scheduler not implemented yet")
            from rbf.prior.denoise_schedulers import GeneralConvexScheduler
            self.new_scheduler = GeneralConvexScheduler(
                n=self.cfg.scheduler_n,
                device=self.pipeline.device,
            )

            print_note(f"[***] Using scheduler conversion to {self.new_scheduler.__class__.__name__}")


        elif self.cfg.convert_scheduler == "ot":
            raise NotImplementedError("OT scheduler not implemented yet")
            # self.new_scheduler = CondOTScheduler(device=self.pipeline.device)

        else:
            print(f"[***] Not using scheduler conversion")


    @property
    def rgb_res(self):
        return 1, 3, 512, 512
    
    @property
    def latent_res(self):
        return 1, 4, 64, 64
    
    def prepare_cond(
        self, 
        text_prompt=None, 
        negative_prompt=None,
        _pass=False,
    ):

        if not _pass:
            if hasattr(self, "cond"):
                return self.cond
        
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        print_info("Encoding text prompt", text_prompt)

        text_embeddings = self.encode_text(
            text_prompt, negative_prompt
        )

        neg, pos = text_embeddings.chunk(2)

        self.cond = {
            "neg": neg,
            "pos": pos, 
        }

        # Edited 2025.01.27
        # Save memory
        self.pipeline.text_encoder.to("cpu")

        return self.cond


    def encode_text(self, prompt, negative_prompt=None):
        """
        Encode a text prompt into a feature vector.
        """
        assert self.pipeline is not None, "Pipeline not initialized"

        text_embeddings = self.pipeline.encode_prompt(
            prompt,
            self.cfg.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
        )

        # uncond, cond
        text_embeddings = torch.cat([text_embeddings[1], text_embeddings[0]])

        return text_embeddings
    

    def sample(self, camera, text_prompt=None):
        if text_prompt is None:
            text_prompt = self.cfg.text_prompt

        with torch.no_grad():
            images = self.pipeline(
                prompt=text_prompt, 
                output_type="latent"
            )

        return images
    

    def fast_sample(
        self, camera, x_t, 
        timesteps, guidance_scale=None, 
        text_prompt=None, negative_prompt=None
    ):
        
        return self.sample(camera, text_prompt)
    

    def init_latent(self, batch_size, latents=None):
        num_images_per_prompt = 1
        num_channels_latents = self.pipeline.unet.config.in_channels

        latents = self.pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            self.cfg.height,
            self.cfg.width,
            self.pipeline.dtype,
            self.pipeline.device,
            generator=None,
            latents=None,
        )

        return latents
        

    def predict(
        self, 
        x_t, 
        timestep, 
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
                cfg_cur_x_t_batch, 
                cfg_cur_t,
                **cur_cond,
            ).sample

            cur_noise_pred_uncond, cur_noise_pred_text = cur_noise_pred.chunk(2)
            cur_noise_pred = cur_noise_pred_uncond + self.cfg.guidance_scale * (
                cur_noise_pred_text - cur_noise_pred_uncond
            )

            self.nfe += len(cfg_cur_x_t_batch)
            noise_pred.append(cur_noise_pred)

        noise_pred = torch.cat(noise_pred, dim=0)

        # latent_model_input = torch.cat([x_t] * 2)
        # vec_t = torch.ones((latent_model_input.shape[0],), device=x_t.device) * timestep # [1000 - 0]

        # v_pred = self.pipeline.unet(
        #     latent_model_input, 
        #     vec_t, 
        #     **self.cond,
        # ).sample

        # v_pred_neg, v_pred_text = v_pred.chunk(2)
        # v_pred = v_pred_neg + self.cfg.guidance_scale * (v_pred_text - v_pred_neg)

        if return_dict:
            return {
                "noise_pred": noise_pred,
            }
        
        return noise_pred

    
    # ================================================================
    # Implemented in the parent class
    # ================================================================
    # def encode_image(self, img_tensor):
    #     pass 
    
    # def decode_latent(self, latent, convert_to_float=True):
    #     pass
    
    # def decode_latent_fast(self, latent):
    #     pass
    
    # def decode_latent_fast_if_needed(self, latent):
    #     pass

    # def encode_image_if_needed(self, img_tensor):
    #     if img_tensor.shape[1] == 3:
    #         return self.encode_image(img_tensor)
    #     return img_tensor
    
    # def decode_latent_if_needed(self, latent):
    #     if latent.shape[1] == 4:
    #         return self.decode_latent(latent)
    #     return latent

    # ================================================================
    
    
    def step(
        self, 
        x, 
        t_curr,
        d_t,
        model_pred=None, 
        prev_timestep=None,
    ):
        # InstaFlow: +velocity (reverse)
        _model_pred = -1 * model_pred

        if self.cfg.sample_method in ["ode", "sde"]:
            assert _model_pred is not None, "Model prediction not provided"
            assert torch.all(d_t >= 0.0).item() and torch.all(d_t <= 1.0).item(), f"Invalid time step {d_t}"

            diffuse = self.pipeline.scheduler.get_diffuse(
                t_curr,
                self.cfg.sample_method, 
                self.cfg.diffusion_coefficient, 
                self.cfg.diffusion_norm,
                self.cfg.convert_scheduler,
                new_scheduler=self.new_scheduler,
                original_scheduler=self.original_scheduler,
            )

            drift = sm.prior.pipeline.scheduler.get_drift(
                x, 
                model_pred, 
                t_curr, 
                self.cfg.sample_method, 
                diffusion_coefficient=self.cfg.diffusion_coefficient, 
                diffusion_norm=self.cfg.diffusion_norm,
                convert_scheduler=self.cfg.convert_scheduler,
                new_scheduler=self.new_scheduler,
                original_scheduler=self.original_scheduler,
                diffuse=diffuse, # EDITED
                # diffuse=batch_diffuse,
            )

            prev_x_mean = x + drift * d_t

            w = torch.randn(x.size()).to(x)
            dw = w * torch.sqrt(torch.abs(d_t))

            prev_latent = prev_x_mean + diffuse * dw


            # -------------- ODE/SDE Variance Test -----------------
            
            # with torch.no_grad():
            #     if (self.cfg.batch_size == 1) and (not self.cfg.disable_debug) and (step % self.cfg.log_interval == 0):
            #         camera = sm.dataset.generate_sample()
            #         x0s_logs = []

            #         for _i in range(8):
            #             w = torch.randn(x.size()).to(x)
            #             dw = w * torch.sqrt(torch.abs(dt))

            #             _prev_latent = prev_x_mean + diffuse * dw

            #             if self.cfg.convert_scheduler is not None:
            #                 _model_pred_logging = sm.prior.compute_velocity_transform_scheduler(camera, _prev_latent, prev_timestep)
            #             else:
            #                 _model_pred_logging = sm.prior.predict(camera, _prev_latent, prev_timestep)

            #             _x0 = sm.prior.get_tweedie(_prev_latent, _model_pred_logging, prev_timestep, step)

            #             x0s_logs.append(sm.prior.decode_latent(_x0))

            # -------------------------------------------------------

        
        # -------------- ODE/SDE Variance Logging -----------------

        # if (self.cfg.batch_size == 1) and (not self.cfg.disable_debug) and (step % self.cfg.log_interval == 0):            
        #     imgs_logs = torch_to_pil_batch(torch.cat(x0s_logs), is_grayscale=False)
            
        #     grid_img = image_grid(imgs_logs, 2, len(imgs_logs) // 2).resize((1024, 512))
        #     grid_img.save(
        #         os.path.join(sm.logger.debug_dir, f"{self.cfg.sample_method}_variance_test_{int(prev_timestep)}.png")
        #     )
        
        # --------------------------------------------------------------------


        return prev_latent
    
    

    def get_tweedie(self, noisy_sample, model_pred, t, step):
        # InstaFlow: +velocity (reverse)
        # Flux: -velocity (reverse)
        _model_pred = -1 * model_pred

        timestep = torch.tensor(t / 1000.0).to(_model_pred)
        if self.cfg.convert_scheduler is not None:
            print_note("[***] Using converted scheduler in computing Tweedies")
            cur_scheduler = self.new_scheduler
        else:
            print("[*] Not using the converted scheduler in computing Tweedies")
            cur_scheduler = self.original_scheduler

        r_scheduler_output = cur_scheduler(t=timestep.float())

        alpha_r = r_scheduler_output.alpha_t.to(_model_pred.dtype)
        sigma_r = r_scheduler_output.sigma_t.to(_model_pred.dtype)
        d_alpha_r = r_scheduler_output.d_alpha_t.to(_model_pred.dtype)
        d_sigma_r = r_scheduler_output.d_sigma_t.to(_model_pred.dtype)

        numer = (sigma_r * _model_pred) - (d_sigma_r * noisy_sample)
        denom = (d_alpha_r * sigma_r) - (d_sigma_r * alpha_r)

        return numer / denom
    

    def get_eps(self, noisy_sample, tweedie, t):
        raise NotImplementedError("Eps not supported")
    

    
    def compute_gradient(self, camera, noisy_sample, timestep, guidance_scale, text_prompt=None, negative_prompt=None):
        raise NotImplementedError(f"Gradient computation not supported")
    
        sigmas = (1 - self.ddim_scheduler.alphas_cumprod) ** (0.5)
        sigma = sigmas[timestep]

        noise_preds = self.predict(
            camera, noisy_sample, timestep, 
            guidance_scale=guidance_scale, return_dict=False,
            text_prompt=text_prompt, negative_prompt=negative_prompt,
        )
        score_preds = -noise_preds / sigma

        return score_preds

    def get_noisy_sample(
        self, pred_original_sample, 
        eps, t, eta=0, t_next=None, noise=None
    ):
    
        raise NotImplementedError("Noisy sample not supported")
        

    def move_step(
        self, sample, denoise_eps, 
        src_t, tgt_t, eta=0, renoise_eps=None
    ):  
        assert renoise_eps is not None, "Renoise eps not provided"

        pred_original_sample = self.get_tweedie(
            sample, denoise_eps, src_t
        )

        next_sample = self.get_noisy_sample(
            pred_original_sample, renoise_eps, 
            tgt_t, eta=eta, t_next=src_t
        )

        return next_sample
    

    def accelerate_ode(
        self,
        camera, 
        latent_noisy, 
        src_t, tgt_t, 
        num_steps=4, 
        try_fast=True,
        **kwargs,
    ):
        raise NotImplementedError("ODE acceleration not supported")
        
        # if self.cfg.convert_scheduler == "vp":
        #     print_note("[***] Rolling back to CondOT to accelerate ODE")
        #     # Timestep for the original scheduler 
        #     src_t = self.pipeline.schedulerget_new_timestep(
        #         src_t, self.original_scheduler, self.new_scheduler
        #     )
        
        #     # TODO: Roll back VP to CondOT
        
        # x0 = self.ddim_loop(
        #     camera, latent_noisy, src_t, 0, 
        #     num_steps=self.cfg.ode_steps, 
        #     try_fast=self.cfg.try_fast_sampling,
        # )

        # return (x0, x0) # Treat x_t same as x0


    @torch.no_grad()
    def ddim_loop(
        self,
        camera,
        x_t,
        src_t,
        tgt_t,
        mode="cfg",
        guidance_scale=None,
        inv_guidance_scale=None,
        eta=0,
        num_steps=30,
        edge_preserve=False,
        clean=None,
        soft_mask=None,
        sdi_inv=False,
        try_fast=True,
        **kwargs,
    ):
        
        if isinstance(src_t, torch.Tensor):
            src_t = src_t.item()
        if isinstance(tgt_t, torch.Tensor):
            tgt_t = tgt_t.item()

        x_t = x_t.detach().to(self.dtype)

        # linearly interpolate between 1000 and 0
        raw_timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long)

        if src_t == tgt_t:
            return x_t
        
        elif src_t < tgt_t: # inversion
            timesteps = reversed(raw_timesteps)
            from_idx = torch.where(timesteps > src_t)[0]
            from_idx = from_idx[0] if len(from_idx) > 0 else len(timesteps)
            to_idx = torch.where(timesteps < tgt_t)[0]
            to_idx = to_idx[-1] if len(to_idx) > 0 else -1
            timesteps = torch.cat(
                [
                    torch.tensor([src_t]),
                    timesteps[from_idx : to_idx + 1],
                    torch.tensor([tgt_t]),
                ]
            )
            _sigmas = timesteps / 1000
            d_sigma = _sigmas[0] - _sigmas[1]

        elif src_t > tgt_t: # generation
            timesteps = raw_timesteps
            from_idx = torch.where(timesteps < src_t)[0]
            from_idx = from_idx[0] if len(from_idx) > 0 else len(timesteps)
            to_idx = torch.where(timesteps > tgt_t)[0]
            to_idx = to_idx[-1] if len(to_idx) > 0 else -1
            timesteps = torch.cat(
                [
                    torch.tensor([src_t]),
                    timesteps[from_idx : to_idx + 1],
                    torch.tensor([tgt_t]),
                ]
            )

        _sigmas = timesteps / 1000
        # print("Running ddim loop with timesteps", timesteps[:-1])
        # print("_sigmas", _sigmas)
        for i, t_curr in enumerate(timesteps[:-1]):
            model_output = self.predict(
                camera, x_t, t_curr, 
                return_dict=False, **kwargs
            )
            sample = x_t.to(torch.float32)

            d_sigma = -(_sigmas[i] - _sigmas[i+1])
            prev_sample = sample + d_sigma * model_output

            x_t = prev_sample.to(model_output.dtype)

        return x_t


    def compute_velocity_transform_scheduler(self, camera, x, t, **extras):
        t = torch.tensor(t / 1000.0).to(x).to(torch.float32)
        r = t
        
        # print("Current r", r.item()) # Timestep that we want to sample at
        r_scheduler_output = self.new_scheduler(t=r)

        alpha_r = r_scheduler_output.alpha_t
        sigma_r = r_scheduler_output.sigma_t
        d_alpha_r = r_scheduler_output.d_alpha_t
        d_sigma_r = r_scheduler_output.d_sigma_t
        # print("Current", "alpha_r", alpha_r.item(), "sigma_r", sigma_r.item(), \
        #       "d_alpha_r", d_alpha_r.item(), "d_sigma_r", d_sigma_r.item())

        t = self.original_scheduler.snr_inverse(alpha_r / sigma_r)

        # print("New t", t.item()) # Adjusted timestep that we have access to the model
        t_scheduler_output = self.original_scheduler(t=t)

        alpha_t = t_scheduler_output.alpha_t
        sigma_t = t_scheduler_output.sigma_t
        d_alpha_t = t_scheduler_output.d_alpha_t
        d_sigma_t = t_scheduler_output.d_sigma_t
        # print("New", "alpha_t", alpha_t.item(), "sigma_t", sigma_t.item(), \
        #       "d_alpha_t", d_alpha_t.item(), "d_sigma_t", d_sigma_t.item())

        s_r = sigma_r / sigma_t

        dt_r = (
            sigma_t
            * sigma_t
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
        )

        ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

        # u_t = self.model(
        #     x=x / s_r, t=t, **extras
        # )

        # print("[*] Computing velocity at", (t * 1000.0).item(), "\n")
        print("Transformed t", t[0].item() * 1000.0, "Scale ", s_r[0].item())
        u_t = self.predict(camera, x / s_r, (t * 1000.0).to(x.dtype))

        u_r = (ds_r * x / s_r + dt_r * s_r * u_t).to(x.dtype)

        return u_r