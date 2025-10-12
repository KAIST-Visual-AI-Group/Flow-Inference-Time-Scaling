import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image 
import numpy as np
from tqdm import tqdm

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


class RBFDPS:
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
        time_schedule: str = "linear"

        batch_size: int = 1
        width: int = 512
        height: int = 512
        t_max: int = 1000
        max_nfe: int = 500

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
        filtering_method: str = None

        img_idx: int = 0
        benchmark: bool = False
        prompt: str = None

        # Ours Pre-correct
        init_n_particles: int = 50

        # Guidance param
        guidance_strength: float = 1.0
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

        assert self.cfg.time_schedule == "linear", "time_schedule must be linear for DPS"

        self.max_budget = int((self.cfg.max_nfe - self.cfg.init_n_particles * self.cfg.batch_size) / (self.cfg.max_steps * self.cfg.batch_size)) * torch.ones(self.cfg.batch_size, dtype=torch.int32, device = self.cfg.device)
        self.default_budget = int((self.cfg.max_nfe - self.cfg.init_n_particles * self.cfg.batch_size) / (self.cfg.max_steps * self.cfg.batch_size))

    def finished(self, nfe, cur_step_best_tweedie, cur_step_best_rewards) -> Any:
        print_info("Max NFE reached!", nfe) if not sm.OFF_LOG else None

        max_idx = torch.argmax(cur_step_best_rewards).item()
        tweedie = cur_step_best_tweedie[max_idx:max_idx+1]
        latest_rewards = cur_step_best_rewards[max_idx].item()
        sm.model.image = tweedie
        return {"done": True, "lastest_rewards": latest_rewards}

    def train_single_step(self, sample_dict: dict) -> Any:
        with torch.enable_grad():
            nfe = sample_dict["nfe"]
            pbar = sample_dict["pbar"]
            t_curr = sample_dict["t_curr"]
            device = sample_dict["device"]
            model_pred = sample_dict.get("model_pred", None)
            best_rewards = sample_dict["best_rewards"]

            cur_step_best_model_pred = sample_dict.get("cur_step_best_model_pred", None)
            cur_step_best_rewards = sample_dict.get("cur_step_best_rewards", None)
            cur_step_best_latent_noisy = sample_dict.get("cur_step_best_latent_noisy", None)
            cur_step_best_tweedie = sample_dict.get("cur_step_best_tweedie", None)

            budget_counter = sample_dict.get("budget_counter", None)
            step_counter = sample_dict.get("step_counter", None)

            # >>> For DPS
            derivative = sample_dict.get("derivative", None)
            cur_step_best_derivative = sample_dict.get("cur_step_best_derivative", None)
            # <<< For DPS
            t_curr_conv = sm.time_sampler(t_query = t_curr)


            if nfe == 0:
                # Flow-based models
                if self.cfg.prior in ["flux", "instaflow"]:
                    latent_noisy = sm.prior.init_latent(
                        self.cfg.batch_size * self.cfg.init_n_particles
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

            if model_pred is None:
                with torch.no_grad():
                    assert nfe == 0, "model_pred must be provided for nfe > 0"
                    t_curr_conv_expanded = t_curr_conv.repeat(self.cfg.init_n_particles, *([1]*(t_curr_conv.dim()-1)))
                    model_pred = sm.prior.compute_velocity_transform_scheduler(
                        latent_noisy, 
                        t_curr_conv_expanded,
                    ) 
                    nfe += latent_noisy.shape[0]
                    pbar.update(latent_noisy.shape[0])
                    
                    assert nfe < self.cfg.max_nfe, f"batch_size ({nfe}) should be smaller than max_nfe({self.cfg.max_nfe})"

                    init_tweedie = sm.prior.get_tweedie(latent_noisy, model_pred, t_curr_conv_expanded)
                    latent_noisy, model_pred, cur_step_best_tweedie, best_rewards = sm.corrector.pre_correct(latent_noisy, init_tweedie, model_pred, nfe)
                    best_rewards = best_rewards.to(device)

                    budget_counter = torch.ones(self.cfg.batch_size, dtype=torch.int32, device = device)
                    step_counter = torch.zeros(self.cfg.batch_size, dtype=torch.int32, device = device)

                    cur_step_best_model_pred = torch.zeros_like(model_pred)
                    cur_step_best_latent_noisy = torch.zeros_like(latent_noisy)
                    
                    # For DPS
                    derivative = torch.zeros_like(latent_noisy)
                    cur_step_best_derivative = torch.zeros_like(derivative)

            assert latent_noisy.shape[0] == self.cfg.batch_size, f"{latent_noisy.shape[0]} != {self.cfg.batch_size}"
            assert model_pred.shape[0] == self.cfg.batch_size, f"{model_pred.shape[0]} != {self.cfg.batch_size}"

            t_prev = torch.clamp(t_curr - self.cfg.t_max / self.cfg.max_steps, min = 0.0)
            t_prev_conv = sm.time_sampler(t_query = t_prev)
            d_t = t_curr_conv - t_prev_conv

            new_latent_noisy = sm.prior.step(
                latent_noisy, # B*N, 4, H, W (x_t)
                t_curr = t_curr_conv, # B
                d_t = d_t / 1000.0,
                model_pred = model_pred, # B*N, 4, H, W (v_t)
                prev_timestep = t_prev_conv, # logging purpose
            ).to(latent_noisy.dtype); # B*N, 4, H, W (x_t-1)

            dnfe = latent_noisy.shape[0]

            if nfe + dnfe >= self.cfg.max_nfe:
                return self.finished(nfe, cur_step_best_tweedie, cur_step_best_rewards)

            # >>> Guidance Tempering
            tempering_factor_list = list()
            for step_ in step_counter:
                tempering_factor = torch.tensor([min((1 + self.cfg.tempering_gamma) ** (step_.item() + 1) - 1, 1.)], device=self.cfg.device, dtype=torch.bfloat16)
                tempering_factor_list.append(tempering_factor)

            tempering_factor = torch.tensor(tempering_factor_list, device=self.cfg.device, dtype=derivative.dtype)
            derivative *= tempering_factor[:, None, None]

            alpht_t_list = list()
            for t_curr_ in t_curr:
                t_curr_coeff = sm.prior.new_scheduler(t=(t_curr_/1000.0))
                alpha_t = t_curr_coeff.alpha_t.item()
                alpht_t_list.append(alpha_t)
            
            alpha_t = torch.tensor(alpht_t_list, device=self.cfg.device, dtype=derivative.dtype)
            # <<< Guidance Tempering

            new_latent_noisy = new_latent_noisy + self.cfg.guidance_strength * derivative * alpha_t[:, None, None]

            if new_latent_noisy.requires_grad:
                new_latent_noisy=new_latent_noisy.detach()
            new_latent_noisy.requires_grad_(True)
            # <<< Guidance

            nfe += dnfe
            with open(os.path.join(sm.logger.debug_dir, "nfe.txt"), "a") as f:
                f.write(f"{nfe} {torch.max(best_rewards).item()}\n")
            pbar.update(dnfe)

            new_model_pred = sm.prior.compute_velocity_transform_scheduler(
                new_latent_noisy, 
                t_prev_conv,
            ).to(latent_noisy.dtype) # B, 4, H, W (v_t-1)

            new_tweedie = sm.prior.get_tweedie(
                new_latent_noisy, # x_t-1 / x_t-1+tau
                new_model_pred, # v_t-1
                t_prev_conv, # t-1
            ) # B*N, 4, H, W (x_0|t-1)

            target = list()
            decoded_tweedie_list = list()
            for __i in range(0, len(new_tweedie), self.cfg.minibatch_size):
                cur_batch_size = min(self.cfg.minibatch_size, len(new_tweedie) - __i)
                decoded_tweedies = sm.prior.decode_latent(new_tweedie[__i:__i+cur_batch_size])
                decoded_tweedie_list.append(decoded_tweedies.clone().detach())
                target += [sm.corrector.reward_model.preprocess(decoded_tweedies)]

            target = torch.cat(target)
            decoded_tweedies_list = torch.cat(decoded_tweedie_list)
            new_rewards = sm.corrector.reward_model(target, nfe, decoded_tweedies_list) * 200.0

            rollover_index = best_rewards < new_rewards
            cur_step_best_update_bool = cur_step_best_rewards < new_rewards
            proceed = step_counter < (self.cfg.max_steps - 2)

            budget_full_index = (budget_counter % self.max_budget) == 0
            budget_full_index = budget_full_index & ~rollover_index
            budget_full_index = budget_full_index & proceed

            if torch.any(cur_step_best_update_bool).item():
                cur_step_best_model_pred[cur_step_best_update_bool] = new_model_pred[cur_step_best_update_bool]
                cur_step_best_rewards[cur_step_best_update_bool] = new_rewards[cur_step_best_update_bool].to(torch.float32)
                cur_step_best_latent_noisy[cur_step_best_update_bool] = new_latent_noisy[cur_step_best_update_bool]
                cur_step_best_tweedie[cur_step_best_update_bool] = new_tweedie[cur_step_best_update_bool]

                # >>> Guidance
                for __i, do_update in enumerate(cur_step_best_update_bool):
                    if do_update:
                        grad_ = torch.autograd.grad(new_rewards[__i], new_latent_noisy, retain_graph=True)[0][__i]
                        grad_ = torch.nan_to_num(grad_)
                        cur_step_best_derivative[__i] = grad_.detach()
                # <<< Guidance

            if torch.any(rollover_index).item():
                best_rewards[rollover_index] = new_rewards[rollover_index].to(torch.float32)
                # ===============================================
                # Logging
                # ===============================================
                if (not self.cfg.disable_debug) and not sm.OFF_LOG:
                    # NOTE: Batch-wise logging
                    print_info(f"Logging at NFE = {nfe}") if not sm.OFF_LOG else None;

                    xts_logs = []
                    x0s_logs = []
                    for _b in range(len(latent_noisy[:min(len(latent_noisy), self.cfg.n_max_log)])):
                        prev_latent_for_log = latent_noisy[_b:_b+1]
                        tweedie_for_log = cur_step_best_tweedie[_b:_b+1].detach()

                        if prev_latent_for_log.shape[1] != 3:
                            prev_latent_for_log = sm.prior.decode_latent(prev_latent_for_log)
                        
                        if tweedie_for_log.shape[1] != 3:
                            tweedie_for_log = sm.prior.decode_latent(tweedie_for_log)

                        xts_logs.append(prev_latent_for_log)
                        x0s_logs.append(tweedie_for_log)

                    pil_x0s = torch_to_pil_batch(torch.cat(x0s_logs), is_grayscale=False)
                    grid_x0 = image_grid(pil_x0s, 1, len(pil_x0s)).resize((1024 * len(pil_x0s), 1024))
                    
                    pil_xts = torch_to_pil_batch(torch.cat(xts_logs), is_grayscale=False)
                    grid_xt = image_grid(pil_xts, 1, len(pil_xts)).resize((1024 * len(pil_xts), 1024))

                    grid_img = image_grid([grid_x0, grid_xt], 2, 1)
                    grid_img.save(os.path.join(sm.logger.training_dir, f"training_{nfe:03d}.png"))

                rollover_proceed = rollover_index & proceed
                if torch.any(rollover_proceed).item():
                    t_curr[rollover_proceed] = t_prev[rollover_proceed]
                    latent_noisy[rollover_proceed] = new_latent_noisy[rollover_proceed]
                    model_pred[rollover_proceed] = new_model_pred[rollover_proceed]

                    leftover = self.max_budget[rollover_proceed] - budget_counter[rollover_proceed]
                    budget_counter[rollover_proceed] = 0
                    step_counter[rollover_proceed] += 1
                    cur_step_best_rewards[rollover_proceed] = -1e9
                    self.max_budget[rollover_proceed] = leftover + self.default_budget

                    # >>> Guidance
                    for __i, do_update in enumerate(rollover_index):
                        if do_update:
                            grad_ = torch.autograd.grad(new_rewards[__i], new_latent_noisy, retain_graph=True)[0][__i]
                            grad_ = torch.nan_to_num(grad_)
                            derivative[__i] = grad_.detach()
                    # <<<  Guidance

            if torch.any(budget_full_index).item():
                t_curr[budget_full_index] = t_prev[budget_full_index]
                latent_noisy[budget_full_index] = cur_step_best_latent_noisy[budget_full_index]
                model_pred[budget_full_index] = cur_step_best_model_pred[budget_full_index]

                budget_counter[budget_full_index] = 0
                step_counter[budget_full_index] += 1
                cur_step_best_rewards[budget_full_index] = -1e9
                self.max_budget[budget_full_index] = self.default_budget

                derivative[budget_full_index] = cur_step_best_derivative[budget_full_index]

                if (not self.cfg.disable_debug) and not sm.OFF_LOG:
                    # NOTE: Batch-wise logging
                    print_info(f"Logging at NFE = {nfe}") if not sm.OFF_LOG else None

                    xts_logs = []
                    x0s_logs = []
                    for _b in range(len(latent_noisy[:min(len(latent_noisy), self.cfg.n_max_log)])):
                        prev_latent_for_log = latent_noisy[_b:_b+1]
                        tweedie_for_log = cur_step_best_tweedie[_b:_b+1].detach()

                        if prev_latent_for_log.shape[1] != 3:
                            prev_latent_for_log = sm.prior.decode_latent(prev_latent_for_log)
                        
                        if tweedie_for_log.shape[1] != 3:
                            tweedie_for_log = sm.prior.decode_latent(tweedie_for_log)

                        xts_logs.append(prev_latent_for_log)
                        x0s_logs.append(tweedie_for_log)

                    pil_x0s = torch_to_pil_batch(torch.cat(x0s_logs), is_grayscale=False)
                    grid_x0 = image_grid(pil_x0s, 1, len(pil_x0s)).resize((256 * len(pil_x0s), 256))
                    
                    pil_xts = torch_to_pil_batch(torch.cat(xts_logs), is_grayscale=False)
                    grid_xt = image_grid(pil_xts, 1, len(pil_xts)).resize((256 * len(pil_xts), 256))

                    # Merge xts/x0s grids
                    grid_img = image_grid([grid_x0, grid_xt], 2, 1)
                    grid_img.save(os.path.join(sm.logger.training_dir, f"training_{nfe:03d}_step.png"))

            sample_dict = {
                "xts": latent_noisy.detach(),
                "model_pred": model_pred.detach(),
                "t_curr": t_curr,
                "pbar": pbar,
                "nfe": nfe,
                "device": device,
                "best_rewards": best_rewards.detach(),
                "done": False,

                "cur_step_best_model_pred": cur_step_best_model_pred.detach(),
                "cur_step_best_rewards": cur_step_best_rewards.detach(),
                "cur_step_best_latent_noisy": cur_step_best_latent_noisy.detach(),
                "cur_step_best_tweedie": cur_step_best_tweedie.detach(),

                "budget_counter": budget_counter + 1,
                "step_counter": step_counter,

                "derivative": derivative.detach(),
                "cur_step_best_derivative": cur_step_best_derivative.detach(),
            }

            return sample_dict


    def train(self):
        device = "cuda:{}".format(self.cfg.device)
        sample_dict = {
            "nfe": 0,
            "device": device,
            "t_curr": torch.full((self.cfg.batch_size, 1, 1), self.cfg.t_max, dtype = torch.float32, device = device),
            "best_rewards": torch.full((self.cfg.batch_size,), -1e9, dtype = torch.float32, device = device),
            "cur_step_best_rewards": torch.full((self.cfg.batch_size,), -1e9, dtype = torch.float32, device = device),
        }
        self.max_budget = int((self.cfg.max_nfe - self.cfg.init_n_particles * self.cfg.batch_size) / (self.cfg.max_steps * self.cfg.batch_size)) * torch.ones(self.cfg.batch_size, dtype=torch.int32, device = device)
        self.default_budget = int((self.cfg.max_nfe - self.cfg.init_n_particles * self.cfg.batch_size) / (self.cfg.max_steps * self.cfg.batch_size))
        
        if not self.cfg.benchmark:
            with redirect_stdout_to_tqdm():
                with re_trange(0, self.cfg.max_nfe,position=0, desc="NFE", initial=0, total=self.cfg.max_nfe,) as pbar:
                    sample_dict["pbar"] = pbar
                    while True:
                        sample_dict = self.train_single_step(sample_dict)
                        if sample_dict["done"]:
                            break
                    pbar.close()

                output_filename = os.path.join(self.cfg.root_dir, self.cfg.output)
                sm.model.save(output_filename)
                if hasattr(sm.model, "render_eval"):
                    print_info("render_eval detected. Rendering the final image...")
                    sm.model.render_eval(self.eval_dir)
                return output_filename
            
        else:
            with tqdm(range(0, self.cfg.max_nfe), desc="Denoising Step", total=self.cfg.max_nfe, dynamic_ncols=True, leave=False) as pbar:
                sample_dict["pbar"] = pbar
                while True:
                    sample_dict = self.train_single_step(sample_dict)
                    if sample_dict["done"]:
                        break
                pbar.close()

            output_filename = os.path.join(self.cfg.root_dir, f"{self.cfg.prompt}_{self.cfg.seed:05d}.png")
            last_tweedie = sm.model.image
            last_tweedie_img = sm.prior.decode_latent(last_tweedie)
            last_tweedie_img = last_tweedie_img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            last_tweedie_img = (last_tweedie_img*255).astype('uint8')
            Image.fromarray(last_tweedie_img).save(output_filename)

