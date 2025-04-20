from abc import ABC, abstractmethod
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionGLIGENPipeline,
)
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
)

from rbf.utils.extra_utils import weak_lru
from rbf.utils.print_utils import print_info, print_warning
from rbf.utils.image_utils import save_tensor


NEGATIVE_PROMPT = (
    "painting, unreal, twisted"
)


class Prior(ABC):
    def __init__(self):
        super().__init__()
        self.pipeline = None

    @property
    @abstractmethod
    def rgb_res(self):
        pass

    @property
    @abstractmethod
    def latent_res(self):
        pass

    @abstractmethod
    def prepare_cond(self, camera):
        return None

    @abstractmethod
    def sample(self):
        """
        Generate images from a text prompt and other conditions.
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Predict the noise for a given latent `x_t` at a specific timestep.
        """
        pass

    @weak_lru(maxsize=10)
    def encode_text(self, prompt, negative_prompt=None):
        """
        Encode a text prompt into a feature vector.
        """
        assert self.pipeline is not None, "Pipeline not initialized"
        text_embeddings = self.pipeline.encode_prompt(
            prompt, "cuda", 1, True, negative_prompt=negative_prompt
        )
        # uncond, cond
        text_embeddings = torch.cat([text_embeddings[1], text_embeddings[0]])
        return text_embeddings

    @property
    def device(self):
        return self.pipeline.device
    
    @property
    def dtype(self):
        return self.pipeline.vae.dtype

    def encode_image(self, img_tensor):
        assert self.pipeline is not None, "Pipeline not initialized"
        vae = self.pipeline.vae
        flag = False
        if img_tensor.dim() == 3:
            flag = True
            img_tensor = img_tensor.unsqueeze(0)
        x = (2 * img_tensor - 1).to(vae.dtype)
        
        y = []
        for i in range(len(x)):
            y.append(vae.encode(x[i:i+1]).latent_dist.sample() * vae.config.scaling_factor)
        
        y = torch.cat(y, dim=0)
        # x = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
        if flag:
            y = y.squeeze(0)
        return y

    def decode_latent(self, latent):
        assert self.pipeline is not None, "Pipeline not initialized"
        vae = self.pipeline.vae
        flag = False
        if latent.dim() == 3:
            flag = True
            latent = latent.unsqueeze(0)
        
        latent = latent.to(vae.dtype)
        
        x = []
        for i in range(len(latent)):
            x.append(vae.decode(latent[i:i+1] / vae.config.scaling_factor).sample)
        x = torch.cat(x, dim=0)
        
        x = (x / 2 + 0.5).clamp(0, 1)
        if flag:
            x = x.squeeze(0)
        return x.to(torch.float32)
    
    def decode_latent_fast(self, latent):
        self.decode_mtx = torch.tensor(
            [
                #   R       G       B
                [0.298, 0.207, 0.208],  # L1
                [0.187, 0.286, 0.173],  # L2
                [-0.158, 0.189, 0.264],  # L3
                [-0.184, -0.271, -0.473],  # L4
            ]
        ).cuda()

        self.post_processor = lambda x: (
            (0.5 * torch.einsum("bchw,cd->bdhw", x, self.decode_mtx) + 0.5).clamp(0, 1)
        )
        
        return self.post_processor(latent)

    def encode_image_if_needed(self, img_tensor):
        if img_tensor.shape[-3] == 3:
            return self.encode_image(img_tensor)
        return img_tensor
    
    def decode_latent_if_needed(self, latent):
        if latent.shape[-3] == 4:
            return self.decode_latent(latent)
        return latent
    
    def decode_latent_fast_if_needed(self, latent):
        if latent.shape[-3] == 4:
            return self.decode_latent_fast(latent)
        return latent

    def add_noise(self, x, t, eps_prev=None, eta=0):
        if x.shape[0] != self.cfg.batch_size:
            assert self.cfg.batch_size > x.shape[0], f"Config batch size {self.cfg.batch_size} must be greater than sample size {x.shape[0]}"
            assert self.cfg.batch_size % x.shape[0] == 0, f"Config batch size {self.cfg.batch_size} must be divisible by sample size {x.shape[0]}"

            x = x.repeat(self.cfg.batch_size // x.shape[0], 1, 1, 1) # Repeat x0

        if eps_prev is None:
            assert 1==2, "eps_prev must be provided."
            assert eta == 0, "eta must be 0 if noise is not provided."
            eps_prev = torch.randn_like(x)

        # DDIM + DDPM
        next_timestep = t + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(x) if t >= 0 else self.scheduler.final_alpha_cumprod
        variance = self.scheduler._get_variance(next_timestep, t)
        std_dev_t = eta * variance ** (0.5)

        pred_sample_direction = (1 - alpha_prod_t - std_dev_t**2) ** (0.5) * eps_prev
        variance_noise = randn_tensor(
            eps_prev.shape, 
            device=eps_prev.device, 
            dtype=eps_prev.dtype, 
        )
        variance = std_dev_t * variance_noise

        noisy_sample = alpha_prod_t**0.5 * x + pred_sample_direction + variance

        return (noisy_sample, x)



    def get_tweedie(self, noisy_sample, eps_pred, t):
        alpha = self.scheduler.alphas_cumprod[t.long()]
        tweedie = (noisy_sample - (1 - alpha) ** 0.5 * eps_pred) / alpha**0.5

        return tweedie

    def get_eps(self, noisy_sample, tweedie, t):
        alpha = self.scheduler.alphas_cumprod[t]
        eps = (noisy_sample - (alpha**0.5) * tweedie) / (1 - alpha) ** 0.5
        return eps
    
    def compute_gradient(self, camera, noisy_sample, timestep, guidance_scale, text_prompt=None, negative_prompt=None):
        sigmas = (1 - self.scheduler.alphas_cumprod) ** (0.5)
        sigma = sigmas[timestep]

        noise_preds = self.predict(
            camera, noisy_sample, timestep, 
            guidance_scale=guidance_scale, return_dict=False,
            text_prompt=text_prompt, negative_prompt=negative_prompt,
        )
        score_preds = -noise_preds / sigma

        return score_preds

    def get_noisy_sample(self, pred_original_sample, eps, t, eta=0, t_next=None, noise=None):
        if t_next is None:
            interval = 1000 // self.scheduler.num_inference_steps
            t_next = min(t + interval, 999)
        
        alpha = self.scheduler.alphas_cumprod[t]
        # alpha_next = self.scheduler.alphas_cumprod[t_next]
        if eta > 0:
            raise NotImplementedError("Eta > 0 not implemented yet.")
            if t_next > t:
                # sigma = eta * ((1 - alpha)/(1 - alpha_next) * (1 - alpha_next/alpha)) ** 0.5
                sigma = eta * self.scheduler._get_variance(t_next, t) ** (0.5)
            else:
                sigma = eta * self.scheduler._get_variance(t, t_next) ** (0.5)
        else:
            sigma = 0

        tweedie_coeff = alpha ** 0.5
        eps_coeff = (1 - alpha - sigma**2) ** 0.5
        noise_coeff = sigma

        noisy_sample = tweedie_coeff * pred_original_sample + eps_coeff * eps
        
        noise = torch.randn_like(eps) if noise is None else noise
        noisy_sample = noisy_sample + noise_coeff * noise

        return noisy_sample

    def move_step(self, sample, denoise_eps, src_t, tgt_t, eta=0, renoise_eps=None):
        renoise_eps = renoise_eps if renoise_eps is not None else denoise_eps

        pred_original_sample = self.get_tweedie(sample, denoise_eps, src_t)
        next_sample = self.get_noisy_sample(pred_original_sample, renoise_eps, tgt_t, eta=eta, t_next=src_t)
        
        return next_sample

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

        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        orig_dtype = x_t.dtype
        x_t = x_t.detach().to(self.dtype)

        # linearly interpolate between 1000 and 0
        raw_timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long)

        if src_t == tgt_t:
            return x_t
        elif src_t < tgt_t:
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
        elif src_t > tgt_t:
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
        
        if try_fast and not edge_preserve and mode == "cfg":
            if hasattr(self, "fast_sample"):
                print_info("Fast sampling enabled")
                output = self.fast_sample(
                    camera, x_t, timesteps[:-1], 
                    guidance_scale=guidance_scale, **kwargs
                )
                return output
        
        if edge_preserve:
            N = len(timesteps - 1)
            H, W = x_t.shape[-2:]
            if soft_mask is not None:
                mask = soft_mask
            else:
                x = torch.linspace(0, 1, W//2)  # from 0 to 1 in 32 steps
                x = torch.cat([x, torch.flip(x, dims=[0])])  # mirror it to go back to 0 (64 steps in total)
                mask = x.repeat(H, 1).to(self.device)
            # make sure that mask is of the same size as the image
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            clean_eps = torch.randn_like(x_t)

        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # print(f"DDIM step {i+1}/{len(timesteps)-1}")

            noise_pred_dict = self.predict(
                camera, x_t, t_curr, guidance_scale=guidance_scale, 
                return_dict=True, **kwargs
            )
            noise_pred, noise_pred_uncond, noise_pred_text = (
                noise_pred_dict["noise_pred"],
                noise_pred_dict["noise_pred_uncond"],
                noise_pred_dict["noise_pred_text"],
            )

            if mode == "cfg":
                renoise_eps = noise_pred
            elif mode == "sds":
                renoise_eps = torch.randn_like(noise_pred)
            elif mode == "cfg++":
                renoise_eps = noise_pred_uncond

            x_t = self.move_step(
                x_t, noise_pred, t_curr, t_next, renoise_eps=renoise_eps, eta=eta
            )
            if edge_preserve:
                M = (mask >= (1 - i / N)).to(x_t.dtype)
                x_t = x_t * M + self.get_noisy_sample(clean, clean_eps, t_next) * (1 - M)
                
            if sdi_inv:
                assert eta == 0, "SDI eta must be 0. It uses inversion eta to only add noise to noisy sample."
                assert t_next > t_curr, f"t_next {t_next} must be greater than t_curr {t_curr} for SDI"
                variance = self.scheduler._get_variance(t_next, t_curr) ** (0.5)
                x_t += 0.3 * torch.randn_like(x_t) * variance
                # print_info(f"sdi_inv with randn noise {variance}")

        return x_t.to(orig_dtype)
