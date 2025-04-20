from abc import ABC, abstractmethod
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import inspect
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel

from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_info, print_note
from rbf.prior.base import Prior, NEGATIVE_PROMPT
from rbf import shared_modules as sm 
from rbf.utils.image_utils import torch_to_pil_batch, image_grid


# NEGATIVE_PROMPT = "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"
NEGATIVE_PROMPT = (
    "low quality, blurry, bad anatomy, disfigured, poorly drawn face"
)

@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting=False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
    ):

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        # Use for score computation
        self.d_sigmas = 1 # d\sigma / dt
        self.d_alphas = -1 # d\alpha / dt


    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.config.num_train_timesteps

        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps.to(device=device)
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

        # dt initialization
        self.alphas = 1 - self.sigmas
        self.d_sigma = self.sigmas[:-1] - self.sigmas[1:] # positive step size
        self.dt = self._sigma_to_t(self.d_sigma)
        

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        print(timestep, schedule_timesteps) if not sm.OFF_LOG else None
        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    
    def convert_velocity_to_score(
        self,
        model_output: torch.FloatTensor, 
        t_curr: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        convert_scheduler=None,
        new_scheduler=None,
        original_scheduler=None,
    ):

        new_t = t_curr / 1000.0

        if convert_scheduler is not None:
            print_note("[***] Using convert scheduler in computing score from velocity") if not sm.OFF_LOG else None
            cur_scheduler = new_scheduler

        else:
            print("Not using convert scheduler in computing score from velocity") if not sm.OFF_LOG else None
            cur_scheduler = original_scheduler

        new_schedule_coeffs = cur_scheduler(new_t)
        
        sigma_t = new_schedule_coeffs.sigma_t
        d_sigma_t = new_schedule_coeffs.d_sigma_t
        alpha_t = new_schedule_coeffs.alpha_t
        d_alpha_t = new_schedule_coeffs.d_alpha_t
        
        reverse_alpha_ratio = alpha_t / d_alpha_t

        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * model_output - sample) / var

        return score

    def get_new_timestep(
        self,
        t,
        original_scheduler,
        new_scheduler,
    ):
        
        r = t / 1000.0
        
        r_scheduler_output = new_scheduler(t=r)

        alpha_r = r_scheduler_output.alpha_t
        sigma_r = r_scheduler_output.sigma_t
        # print("Current", "alpha_r", alpha_r.item(), "sigma_r", sigma_r.item(), \
        #       "d_alpha_r", d_alpha_r.item(), "d_sigma_r", d_sigma_r.item())

        new_t = original_scheduler.snr_inverse(alpha_r / sigma_r)
        print("Current r", r.item(), "-->", "New t", new_t.item()) if not sm.OFF_LOG else None

        return new_t
        
    
    # Diffuse
    def get_sde_diffuse(
        self, 
        t_curr,
        diffusion_coefficient="sigma", 
        diffusion_norm=1, 
        convert_scheduler=None,
        new_scheduler=None,
        original_scheduler=None,
    ):
        
        if convert_scheduler is not None:
            print_note("[***] Convert scheduler", convert_scheduler, "Adding noise", diffusion_coefficient, "norm", diffusion_norm) if not sm.OFF_LOG else None
            cur_scheduler = new_scheduler

        else:
            print("No Convert scheduler", "Adding noise", diffusion_coefficient, "norm", diffusion_norm) if not sm.OFF_LOG else None
            cur_scheduler = original_scheduler

        # new_t = self.timesteps[step] / 1000
        new_t = t_curr / 1000.0

        if diffusion_coefficient == "sigma":
            return diffusion_norm * cur_scheduler(new_t).sigma_t
        
        elif diffusion_coefficient == "constant":
            return diffusion_norm
        
        elif diffusion_coefficient == "sin":
            return diffusion_norm * torch.sin(torch.pi/2 * new_t)
        
        elif diffusion_coefficient == "linear":
            return diffusion_norm * new_t
        
        elif diffusion_coefficient == "square":
            return diffusion_norm * (new_t ** 2.0)

        elif diffusion_coefficient == "clipping":
            # return diffusion_norm * max(0.0, 2.0 * (new_t - 0.5))
            return diffusion_norm * torch.clamp(2.0 * (new_t - 0.5), min=0.0)
        
        elif diffusion_coefficient == "exp":
            return diffusion_norm * torch.exp(-((new_t-1) / sm.prior.cfg.exp_diff_coeff_sigma) ** 2.0)
        
        elif diffusion_coefficient == "tan":
            clamp_new_t = torch.clamp(new_t, min=0, max=0.95)
            return diffusion_norm * torch.tan(torch.pi/2 * clamp_new_t)
        
        else:
            raise NotImplementedError(f"Diffusion coefficient {diffusion_coefficient} not implemented")


    def get_ode_diffuse(self):
        return 0

    def get_diffuse(
        self, 
        t_curr,
        sample_method, 
        diffusion_coefficient="sigma", 
        diffusion_norm=1, 
        convert_scheduler=None,
        new_scheduler=None, 
        original_scheduler=None,
    ):
        
        if sample_method == "ode":
            return self.get_ode_diffuse()
        
        elif sample_method == "sde":
            return self.get_sde_diffuse(
                t_curr = t_curr,
                diffusion_coefficient=diffusion_coefficient, 
                diffusion_norm=diffusion_norm, 
                convert_scheduler=convert_scheduler,
                new_scheduler=new_scheduler,
                original_scheduler=original_scheduler,
            )

        else:
            raise ValueError(f"Unknown sample method {sample_method}")
    
    # Drift
    def get_sde_drift(
        self, 
        velocity, 
        t_curr, 
        sample, 
        sample_method, 
        diffusion_coefficient="sigma", 
        diffusion_norm=1,
        convert_scheduler=None,
        new_scheduler=None,
        original_scheduler=None,
        diffuse=None, 
    ):
        
        if diffuse is None:
            diffuse = self.get_diffuse(
                t_curr,
                sample_method, 
                diffusion_coefficient=diffusion_coefficient, 
                diffusion_norm=diffusion_norm,
                convert_scheduler=convert_scheduler,
                new_scheduler=new_scheduler,
                original_scheduler=original_scheduler,
            )

        score = self.convert_velocity_to_score(
            velocity, t_curr, sample, 
            convert_scheduler=convert_scheduler, 
            new_scheduler=new_scheduler, 
            original_scheduler=original_scheduler,
        )

        drift = -velocity + (0.5 * diffuse ** 2) * score

        return drift
    
    def get_ode_drift(self, velocity):
        return -velocity
        
    def get_drift(
        self, 
        noisy_sample, 
        model_pred, 
        t_curr, 
        sample_method, 
        diffusion_coefficient="sigma", 
        diffusion_norm=1,
        convert_scheduler=None,
        new_scheduler=None,
        original_scheduler=None,
        diffuse=None,
    ):

        if sample_method == "ode":
            drift = self.get_ode_drift(model_pred)

        elif sample_method == "sde":
            # if self.step_index is None:
            #     self._init_step_index(timestep)

            drift = self.get_sde_drift(
                model_pred, 
                t_curr, 
                noisy_sample, 
                sample_method, 
                diffusion_coefficient=diffusion_coefficient, 
                diffusion_norm=diffusion_norm,
                convert_scheduler=convert_scheduler,
                new_scheduler=new_scheduler,
                original_scheduler=original_scheduler,
                diffuse=diffuse, 
            )

        else:
            raise NotImplementedError("Invalid sample method")
        
        return drift
        

    def __len__(self):
        return self.config.num_train_timesteps
    

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


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


class FluxPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        save_vram: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

        for _model in [self.text_encoder, self.text_encoder_2, self.transformer, self.vae]:
            for _k, _v in _model.named_parameters():
                _v.requires_grad_(False)

        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.transformer.eval()

        if save_vram:
            self.vae.enable_tiling()
            self.vae.enable_slicing()
            # self.enable_sequential_cpu_offload()
        

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            ) if not sm.OFF_LOG else None

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            ) if not sm.OFF_LOG else None
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor} but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(
            batch_size, 
            num_channels_latents, 
            height // 2, 
            2, 
            width // 2, 
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

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

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # print("shape", shape) if not sm.OFF_LOG else None
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        print("sigmas", sigmas[:5], ",..., ", sigmas[-5:]) if not sm.OFF_LOG else None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )

        print("timesteps", timesteps) if not sm.OFF_LOG else None
        print("num_inference_steps", num_inference_steps) if not sm.OFF_LOG else None

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        print("Total running step: ", len(timesteps)) if not sm.OFF_LOG else None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # print("noise_pred", noise_pred.shape, noise_pred.dtype) # [1, 4096, 64], float16
                # print("latents", latents.shape, latents.dtype)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        return image
    
    
    @torch.no_grad()
    def run_sde(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        print("Total running step: ", len(timesteps)) if not sm.OFF_LOG else None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype

                # NOTE: Running SDE instead of 
                # latents = self.scheduler.step(
                #     noise_pred, t, latents, return_dict=False
                # )[0]

                latents = self.scheduler.sde_step(
                    noise_pred, t, latents, return_dict=False
                )[0]


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        return image


class FluxPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        device: int = 0
        batch_size: int = 1
        minibatch_size: int = 5
        n_particles: int = 1
        model_name: str = "black-forest-labs/FLUX.1-schnell" # black-forest-labs/FLUX.1-dev
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = NEGATIVE_PROMPT
        width: int = 1024
        height: int = 1024
        guidance_scale: int = 3.5
        root_dir: str = "./results/default"
        max_steps: int = 30

        sample_method: str = "ode"
        diffusion_coefficient: str = "sigma"
        diffusion_norm: float = 1.0
        convert_scheduler: str = None
        scheduler_n: float = None
        tau_norm: float = 0.0
        t_max: float = 1000.0

        disable_debug: bool = False
        log_interval: int = 50
        save_vram: bool = False


        # SoP config
        ode_step: int = None

        # Exp diffusion coefficient param
        exp_diff_coeff_sigma: float = 0.1

        # Negative prompt
        negative_prompt: str = None
        true_cfg_scale: float = 2.5
        do_cfg: bool = False
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        if self.cfg.sample_method == "sde":
            assert self.cfg.diffusion_coefficient is not None
            assert self.cfg.diffusion_norm is not None


        if cfg.prior == "flux":
            print("Using prior model: ", self.cfg.model_name) if not sm.OFF_LOG else None
            self.pipeline = FluxPipeline.from_pretrained(
                self.cfg.model_name,
                torch_dtype=torch.bfloat16,
                save_vram=self.cfg.save_vram,
            )

        elif cfg.prior == "flux_fill":
            print("Using prior model: black-forest-labs/FLUX.1-Fill-dev") if not sm.OFF_LOG else None
            from rbf.prior.flux_fill import FluxFillPipeline

            self.pipeline = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                torch_dtype=torch.bfloat16,
                save_vram=self.cfg.save_vram,
            )
        
        else:
            raise NotImplementedError(f"Prior {cfg.prior} not supported")
        

        self.pipeline = self.pipeline.to(self.cfg.device)
        self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.fast_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.nfe = 0

        print_info(self.cfg.text_prompt) if not sm.OFF_LOG else None
        print_info(f"Sampling method: {self.cfg.sample_method}") if not sm.OFF_LOG else None

        sigmas = np.linspace(1.0, 1 / self.cfg.max_steps, self.cfg.max_steps)
        image_seq_len = 4096 # FIXME: change to cfg
        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.base_image_seq_len,
            self.pipeline.scheduler.config.max_image_seq_len,
            self.pipeline.scheduler.config.base_shift,
            self.pipeline.scheduler.config.max_shift,
        )

        # This can be overriden in time_sampler for custom timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            self.cfg.max_steps,
            self.pipeline.device,
            None,
            sigmas,
            mu=mu,
        )

        from rbf.prior.denoise_schedulers import CondOTScheduler
        self.original_scheduler = CondOTScheduler(device=self.pipeline.device)
        self.new_scheduler = None

        print(f"[*] Original scheduler set to {self.original_scheduler.__class__.__name__}") if not sm.OFF_LOG else None
        if self.cfg.convert_scheduler == "vp":
            from rbf.prior.denoise_schedulers import VPScheduler
            self.new_scheduler = VPScheduler(device=self.pipeline.device)


        elif self.cfg.convert_scheduler == "polynomial":
            from rbf.prior.denoise_schedulers import PolynomialConvexScheduler
            self.new_scheduler = PolynomialConvexScheduler(
                n=self.cfg.scheduler_n,
                device=self.pipeline.device,
            )


        elif self.cfg.convert_scheduler == "general":
            raise NotImplementedError("General scheduler not implemented yet")
            from rbf.prior.denoise_schedulers import GeneralConvexScheduler
            self.new_scheduler = GeneralConvexScheduler(
                n=self.cfg.scheduler_n,
                device=self.pipeline.device,
            )


        elif self.cfg.convert_scheduler == "ot":
            raise NotImplementedError("OT scheduler not implemented yet")
            self.new_scheduler = CondOTScheduler(device=self.pipeline.device)

        else:
            print(f"[***] Not using scheduler conversion") if not sm.OFF_LOG else None

        print_note(f"[***] Using scheduler conversion to {self.new_scheduler.__class__.__name__}") if not sm.OFF_LOG else None


    @property
    def rgb_res(self):
        return 1, 3, 1024, 1024
    
    @property
    def latent_res(self):
        return 1, 4096, 64
    
    def prepare_cond(self, text_prompt=None, negative_prompt=None, _pass=False):
        # To save computation time 
        if not _pass:
            if hasattr(self, "cond"):
                return self.cond
        
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
                
        # >>> 
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipeline.encode_prompt(
            prompt=text_prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=self.pipeline.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )
        assert prompt_embeds.shape[0] == 1, "1"
        # <<< 

        self.cond = {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "txt_ids": text_ids,
        }

        if negative_prompt is not None:
            negative_prompts = [negative_prompt] * (self.cfg.batch_size * self.cfg.n_particles)
            negative_prompt_embeds_list = []
            negative_pooled_prompt_embeds_list = []

            for n_prompt in negative_prompts:
                (
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    _,
                ) = self.encode_prompt(
                    prompt=n_prompt,
                    prompt_2=None,
                    prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    device=self.pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                    lora_scale=None,
                )

                negative_prompt_embeds_list.append(negative_prompt_embeds)
                negative_pooled_prompt_embeds_list.append(negative_pooled_prompt_embeds)

            self.cfg.do_cfg = True

            negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=0)
            negative_pooled_prompt_embeds = torch.cat(negative_pooled_prompt_embeds_list, dim=0)

            self.neg_cond = {
                "encoder_hidden_states": negative_prompt_embeds,
                "pooled_projections": negative_pooled_prompt_embeds,
                "txt_ids": text_ids,
            }

        # Save memory
        # self.pipeline.text_encoder.to("cpu")
        # self.pipeline.text_encoder_2.to("cpu")


    def encode_text(self, text):
        raise NotImplementedError("Text encoding not supported")

    def sample(self, text_prompt=None):
        if text_prompt is None:
            text_prompt = self.cfg.text_prompt

        with torch.no_grad():
            images = self.pipeline(
                prompt=text_prompt, 
                output_type="latent"
            )

        return images
    

    def fast_sample(
        self, x_t, 
        timesteps, guidance_scale=None, 
        text_prompt=None, negative_prompt=None
    ):
        
        return self.sample(text_prompt)
    

    def init_latent(self, batch_size, latents=None):
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipeline.prepare_latents(
            batch_size * 1,
            num_channels_latents,
            self.cfg.height,
            self.cfg.width,
            self.pipeline.dtype,
            self.pipeline.device,
            generator = None,
            latents = latents,
        )

        self.pipeline.latent_image_ids = latent_image_ids

        return latents
        

    def predict(
        self, x_t, timestep, 
        return_dict=False, text_prompt=None, negative_prompt=None,
    ):
    
        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        self.prepare_cond(text_prompt, negative_prompt)

        t = timestep.clone() # No expansion
        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([1], self.cfg.guidance_scale, device=x_t.device, dtype=torch.float32)
            guidance = guidance.expand(x_t.shape[0])
        else:
            guidance = None

        noise_pred = []
        for _i in range(0, len(x_t), self.cfg.minibatch_size):
            cur_batch_size = min(self.cfg.minibatch_size, len(x_t) - _i)
            cur_x_t_batch = x_t[_i:_i+cur_batch_size]
            cur_t = t[_i:_i+cur_batch_size]

            cur_cond = {}
            for k, v in self.cond.items():
                if k in ["encoder_hidden_states", "pooled_projections"]:
                    cur_cond[k] = v.repeat(cur_batch_size, *([1] * (v.dim() - 1)))
                else:
                    cur_cond[k] = v

            if guidance is not None:
                cur_guidance = guidance[_i:_i+cur_batch_size]
            else:
                cur_guidance = guidance
            cur_noise_pred = self.pipeline.transformer(
                hidden_states=cur_x_t_batch,
                timestep=cur_t.view(-1) / 1000, # [MB]
                guidance=cur_guidance,
                img_ids=self.pipeline.latent_image_ids, 
                joint_attention_kwargs=None,
                return_dict=False,
                **cur_cond,
            )[0]

            if self.cfg.do_cfg:
                assert self.cfg.negative_prompt is not None, "Negative prompt not provided"
                
                cur_neg_cond = {}
                for k, v in self.neg_cond.items():
                    if k in ["encoder_hidden_states", "pooled_projections"]:
                        cur_neg_cond[k] = v[_i:_i+cur_batch_size]
                    else:
                        cur_neg_cond[k] = v

                cur_neg_noise_pred = self.pipeline.transformer(
                    hidden_states=cur_x_t_batch,
                    timestep=cur_t.view(-1) / 1000, # [MB]
                    guidance=cur_guidance,
                    img_ids=self.pipeline.latent_image_ids, 
                    joint_attention_kwargs=None,
                    return_dict=False,
                    **cur_neg_cond,
                )[0]

                cur_noise_pred = cur_neg_noise_pred + self.cfg.true_cfg_scale * (cur_noise_pred - cur_neg_noise_pred)

            self.nfe += len(cur_x_t_batch)
            noise_pred.append(cur_noise_pred)

        noise_pred = torch.cat(noise_pred, dim=0)

        if return_dict:
            return {
                "noise_pred": noise_pred,
            }
        
        return noise_pred

    
    def encode_image(self, img_tensor):
        assert self.pipeline is not None, "Pipeline not initialized"

        # Variable res setup
        height = self.cfg.height
        width = self.cfg.width
        latent_height = int(height) // self.pipeline.vae_scale_factor
        latent_width = int(width) // self.pipeline.vae_scale_factor
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4

        vae = self.pipeline.vae
        flag = False
        if img_tensor.dim() == 3:
            flag = True
            img_tensor = img_tensor.unsqueeze(0)
        x = (2 * img_tensor - 1).to(vae.dtype)
        
        y = []
        for _i in range(0, len(x), self.cfg.minibatch_size):
            cur_batch_size = min(self.cfg.minibatch_size, len(x) - _i)
            latent = vae.encode(x[_i:_i+cur_batch_size], return_dict=False)[0].mode()
        
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
            latent = self.pipeline._pack_latents(
                latent, cur_batch_size, num_channels_latents, 
                latent_height, latent_width
            )
            y.append(latent)
        
        y = torch.cat(y, dim=0)
        if flag:
            y = y.squeeze(0)
        return y
    

    def decode_latent(self, latent, convert_to_float=True):
        assert self.pipeline is not None, "Pipeline not initialized"

        # Variable res setup
        height = self.cfg.height
        width = self.cfg.width

        vae = self.pipeline.vae
        flag = False
        if latent.dim() == 2:
            flag = True
            latent = latent.unsqueeze(0)
        
        x = []
        for _i in range(0, len(latent), self.cfg.minibatch_size):
            cur_batch_size = min(self.cfg.minibatch_size, len(latent) - _i)
            image = self.pipeline._unpack_latents(
                latent[_i:_i+cur_batch_size], height, width, self.pipeline.vae_scale_factor
            )
            image = (image / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
            image = self.pipeline.vae.decode(image, return_dict=False)[0]
            x.append(image)

        x = torch.cat(x, dim=0)
        
        x = (x / 2 + 0.5).clamp(0, 1)
        if flag:
            x = x.squeeze(0)

        if convert_to_float:
            return x.to(torch.float32)
            
        return x
    
    def decode_latent_fast(self, latent):
        raise NotImplementedError("Fast decoding not supported")

    def encode_image_if_needed(self, img_tensor):
        if img_tensor.shape[-3] == 3:
            return self.encode_image(img_tensor)
        return img_tensor
    
    def decode_latent_if_needed(self, latent):
        if latent.shape[-2] == 4096:
            return self.decode_latent(latent)
        return latent
    
    def decode_latent_fast_if_needed(self, latent):
        raise NotImplementedError("Fast decoding not supported")
    
    def step(
        self, 
        x, 
        t_curr,
        d_t,
        model_pred=None, 
        prev_timestep=None,
    ):
        if self.cfg.sample_method in ["ode", "sde"]:
            assert model_pred is not None, "Model prediction not provided"
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

            # batch_diffuse = diffuse[:, None, None]
            # print("batch_diffuse", batch_diffuse.shape, batch_diffuse)

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
                diffuse=diffuse, 
            )
            
            prev_x_mean = x + drift * d_t

            w = torch.randn(x.size()).to(x)
            dw = w * torch.sqrt(torch.abs(d_t))

            prev_latent = prev_x_mean + diffuse * dw


            # -------------- ODE/SDE Variance Test -----------------
            
        #     with torch.no_grad():
        #         if (self.cfg.batch_size == 1) and (not self.cfg.disable_debug):
        #             x0s_logs = []

        #             for _i in range(8):
        #                 w = torch.randn(x.size()).to(x)
        #                 dw = w * torch.sqrt(torch.abs(d_t))

        #                 _prev_latent = prev_x_mean + diffuse * dw

        #                 if self.cfg.convert_scheduler == "vp":
        #                     _model_pred = sm.prior.compute_velocity_transform_scheduler(_prev_latent, prev_timestep)
        #                 else:
        #                     _model_pred = sm.prior.predict(_prev_latent, prev_timestep)

        #                 _x0 = sm.prior.get_tweedie(_prev_latent, _model_pred, prev_timestep)

        #                 x0s_logs.append(sm.prior.decode_latent(_x0))

        # if (self.cfg.batch_size == 1) and (not self.cfg.disable_debug):
        #     imgs_logs = torch_to_pil_batch(torch.cat(x0s_logs), is_grayscale=False)
            
        #     grid_img = image_grid(imgs_logs, 2, len(imgs_logs) // 2).resize((1024, 512))
        #     grid_img.save(
        #         os.path.join(sm.logger.debug_dir, f"{self.cfg.sample_method}_variance_test_{int(prev_timestep[0])}.png")
        #     )
        
        # --------------------------------------------------------------------


        return prev_latent
    
    

    def get_tweedie(self, noisy_sample, model_pred, t):
        timestep = (t / 1000.0).to(model_pred)
        if self.cfg.convert_scheduler is not None:
            print_note("[***] Using converted scheduler in computing Tweedies") if not sm.OFF_LOG else None
            cur_scheduler = self.new_scheduler
        else:
            print("[*] Not using the converted scheduler in computing Tweedies") if not sm.OFF_LOG else None
            cur_scheduler = self.original_scheduler

        r_scheduler_output = cur_scheduler(t=timestep.float())

        alpha_r = r_scheduler_output.alpha_t.to(model_pred.dtype)
        sigma_r = r_scheduler_output.sigma_t.to(model_pred.dtype)
        d_alpha_r = r_scheduler_output.d_alpha_t.to(model_pred.dtype)
        d_sigma_r = r_scheduler_output.d_sigma_t.to(model_pred.dtype)

        numer = (sigma_r * model_pred) - (d_sigma_r * noisy_sample)
        denom = (d_alpha_r * sigma_r) - (d_sigma_r * alpha_r)

        return numer / denom


    def compute_velocity_transform_scheduler(self, x, t, **extras):
        print("Original t", t[0].item()) 
        t = (t / 1000.0).to(x).to(torch.float32)
        r = t.clone()
        
        r_scheduler_output = self.new_scheduler(t=r)

        alpha_r = r_scheduler_output.alpha_t
        sigma_r = r_scheduler_output.sigma_t
        d_alpha_r = r_scheduler_output.d_alpha_t
        d_sigma_r = r_scheduler_output.d_sigma_t

        t = self.original_scheduler.snr_inverse(alpha_r / sigma_r)

        t_scheduler_output = self.original_scheduler(t=t)

        alpha_t = t_scheduler_output.alpha_t
        sigma_t = t_scheduler_output.sigma_t
        d_alpha_t = t_scheduler_output.d_alpha_t
        d_sigma_t = t_scheduler_output.d_sigma_t

        s_r = sigma_r / sigma_t

        dt_r = (
            sigma_t
            * sigma_t
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
        )

        ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

        u_t = self.predict((x / s_r).to(x.dtype), (t * 1000.0).to(x.dtype))
        
        u_r = (ds_r * x / s_r + dt_r * s_r * u_t).to(x.dtype)

        return u_r



    def solve_ode(
        self, 
        latent_noisy,
        target_t=0,
        source_t=None, 
        return_tweedie=False, 
    ):
        
        if source_t is None:
            source_t = float(self.cfg.t_max)
            t_curr = torch.full((latent_noisy.shape[0],), source_t).to(self.device).view(-1, 1, 1)

        else:
            t_curr = source_t.clone()

        ode_step = self.cfg.ode_step

        for _ in range(ode_step):
            t_prev = torch.clamp(t_curr - (source_t - target_t) / float(ode_step), min=0.0)

            d_t = t_curr - t_prev

            model_pred = self.compute_velocity_transform_scheduler(
                latent_noisy, t_curr
            )

            tweedie = self.get_tweedie(latent_noisy, model_pred, t_curr)
            t_curr = t_prev

            latent_noisy = latent_noisy - (model_pred * (d_t / 1000.0).to(latent_noisy.dtype))

        if return_tweedie:
            return latent_noisy, tweedie

        return latent_noisy

    