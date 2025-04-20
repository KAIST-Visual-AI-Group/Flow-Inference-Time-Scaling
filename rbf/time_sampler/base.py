from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import pi, sin
import numpy as np 

from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_info
from rbf import shared_modules as sm

class TimeSampler(ABC):
    @ignore_kwargs
    @dataclass
    class Config:
        max_steps: int = 10000
        t_min: int = 20
        t_max: int = 980
        batch_size: int = 10 # batch size
        n_particles: int = 1 # number of particles

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)

    @abstractmethod
    def __call__(self, step):
        pass


class LinearAnnealingTimeSampler(TimeSampler):
    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)

    def __call__(self, step):
        ratio = step / self.cfg.max_steps  # 0.0 ~ 1.0
        t_curr = int(self.cfg.t_max + (self.cfg.t_min - self.cfg.t_max) * ratio)
        t_curr = max(0, min(999, t_curr))

        return t_curr
    

class SDTimeSampler(TimeSampler):
    @ignore_kwargs
    @dataclass
    class Config(TimeSampler.Config):
        max_steps: int = 50
        time_schedule: str = "linear"

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        self.timesteps = None

    def __call__(self, step = None, t_query = None):
        if step is not None:
            if self.timesteps is None:
                import torch 

                timesteps = (
                    np.linspace(0, 1000 - 1, self.cfg.max_steps)
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )

                self.timesteps = torch.from_numpy(timesteps).to(sm.prior.pipeline.device)
                timesteps = torch.cat([self.timesteps, torch.zeros(1, device=sm.prior.pipeline.device)])
                self.d_t = timesteps[:-1] - timesteps[1:]

                print_info(f"Timesteps re-initialized for linear. SD only supports linear", timesteps) if not sm.OFF_LOG else None
            
            batch_time = self.timesteps[step].repeat(self.cfg.batch_size * self.cfg.n_particles, 1, 1, 1)
            batch_d_t = self.d_t[step].repeat(self.cfg.batch_size * self.cfg.n_particles, 1, 1, 1)

            return batch_time, batch_d_t

        else:
            assert t_query is not None;
            if self.cfg.time_schedule == "linear":
                return t_query;
        
            elif self.cfg.time_schedule == "exp":
                return 1000.0 * (1.0 - (1.0 - t_query / 1000.0) ** 2) ** 0.5;



class FluxTimeSampler(TimeSampler):
    @ignore_kwargs
    @dataclass
    class Config(TimeSampler.Config):
        time_schedule: str = "linear"
        max_steps: int = 4
        convert_scheduler: str = None
        scheduler_n: float = None

        prior: str = "flux"

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        self.image_seq_len = 4096 # FIXME: change to cfg
        self.timesteps = None

    def __call__(self, step = None, t_query = None):
        if self.cfg.prior == "sd2":
            if step == len(self.timesteps) - 1:
                return self.timesteps[step], self.timesteps[step]
            
            next_t = self.timesteps[step+1]
            cur_t = self.timesteps[step]
            dt = next_t - self.timesteps[step]
            return cur_t, dt
        
        if step is not None:
            if self.timesteps is None:
                if self.cfg.time_schedule == "linear":
                    self.sigmas = np.linspace(1.0, 1 / self.cfg.max_steps, self.cfg.max_steps)
                    
                elif self.cfg.time_schedule == "exp":
                    _x = np.linspace(0.0, 1.0 - 1 / self.cfg.max_steps, self.cfg.max_steps)
                    self.sigmas = (1-_x **2) ** 0.5
                else:
                    raise NotImplementedError("Unknown flux_time_schedule")

                from rbf.prior.flux import calculate_shift, retrieve_timesteps
                mu = calculate_shift(
                    self.image_seq_len,
                    sm.prior.pipeline.scheduler.config.base_image_seq_len,
                    sm.prior.pipeline.scheduler.config.max_image_seq_len,
                    sm.prior.pipeline.scheduler.config.base_shift,
                    sm.prior.pipeline.scheduler.config.max_shift,
                )

                print("sm.prior.pipeline.device", sm.prior.pipeline.device) if not sm.OFF_LOG else None

                # This can be overriden in time_sampler for custom timesteps
                self.timesteps, _ = retrieve_timesteps(
                    sm.prior.pipeline.scheduler,
                    self.cfg.max_steps,
                    sm.prior.pipeline.device,
                    None,
                    self.sigmas,
                    mu=mu,
                )
                self.d_t = sm.prior.pipeline.scheduler.dt

                print_info(f"Timesteps re-initialized for custom timesteps {self.cfg.time_schedule} ", self.timesteps) if not sm.OFF_LOG else None
            
            batch_time = self.timesteps[step].repeat(self.cfg.batch_size * self.cfg.n_particles, 1, 1)
            batch_d_t = self.d_t[step].repeat(self.cfg.batch_size * self.cfg.n_particles, 1, 1)

            return batch_time, batch_d_t

        else:
            assert t_query is not None;
            if self.cfg.time_schedule == "linear":
                return t_query;
        
            elif self.cfg.time_schedule == "exp":
                return 1000.0 * (1.0 - (1.0 - t_query / 1000.0) ** 2) ** 0.5;


class InstaFlowTimeSampler(TimeSampler):
    @ignore_kwargs
    @dataclass
    class Config(TimeSampler.Config):
        time_schedule: str = "linear"
        max_steps: int = 4

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        self.timesteps = None

    def __call__(self, step):
        if self.timesteps is None:
            if self.cfg.time_schedule == "linear":
                self.sigmas = np.linspace(1.0, 1 / self.cfg.max_steps, self.cfg.max_steps)
                
            elif self.cfg.time_schedule == "exp":
                _x = np.linspace(0.0, 1.0 - 1 / self.cfg.max_steps, self.cfg.max_steps)
                self.sigmas = (1-_x **2) ** 0.5
            else:
                raise NotImplementedError("Unknown flux_time_schedule")

            from rbf.prior.flux import calculate_shift, retrieve_timesteps

            # This can be overriden in time_sampler for custom timesteps
            self.timesteps, _ = retrieve_timesteps(
                sm.prior.pipeline.scheduler,
                self.cfg.max_steps,
                sm.prior.pipeline.device,
                None,
                self.sigmas,
                # mu=mu,
            )

            print_info(f"Timesteps re-initialized for custom timesteps {self.cfg.time_schedule} ", self.timesteps)
        
        return self.timesteps[step].item()
    

