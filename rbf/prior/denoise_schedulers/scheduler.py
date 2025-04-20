# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union

import torch

from torch import Tensor


@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Tensor): :math:`\alpha_t`, shape (...).
        sigma_t (Tensor): :math:`\sigma_t`, shape (...).
        d_alpha_t (Tensor): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (Tensor): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).

    """

    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class Scheduler(ABC):
    """Base Scheduler class."""

    def __init__(self, device=0):
        self.device = device

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""
        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...


class ConvexScheduler(Scheduler):
    def __init__(self, device=0):
        super().__init__(device=device)

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        """Scheduler for convex paths.

        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        """
        Computes :math:`t` from :math:`\kappa_t`.

        Args:
            kappa (Tensor): :math:`\kappa`, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...

    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        
        kappa_t = 1 / (snr + 1.0)

        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """CondOT Scheduler."""
    def __init__(self, device=0):
        super().__init__(device=device)


    def __call__(self, t: Tensor) -> SchedulerOutput:
        
        # assert t >= 0.0 and t <= 1.0, f"t must be in [0,1]. Got t={t}."

        return SchedulerOutput(
            alpha_t=1 - t, # T->0
            sigma_t=t,
            d_alpha_t=-torch.ones_like(t),
            d_sigma_t=torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return kappa


class PolynomialConvexScheduler(ConvexScheduler):
    """Polynomial Scheduler."""

    def __init__(self, n: Union[float, int], device=0) -> None:
        super().__init__(device=device)

        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."

        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        
        assert torch.all(t >= 0.0).item() and torch.all(t <= 1.0).item(), f"t must be in [0,1]. Got t={t}."

        return SchedulerOutput(
            alpha_t=(1 - t)**self.n,
            sigma_t=1- (1 - t)**self.n,
            d_alpha_t=-self.n * ((1 - t) ** (self.n - 1)),
            d_sigma_t=self.n * ((1 - t) ** (self.n - 1)),
        )
    

    def snr_inverse(self, snr: Tensor) -> Tensor:
        kappa_t = 1.0 - (snr / (snr + 1.0)) ** (1.0 / self.n);

        return self.kappa_inverse(kappa=kappa_t)
    

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return kappa
    


class GeneralConvexScheduler(ConvexScheduler):
    """General Scheduler."""

    def __init__(self, n: Union[float, int], device=0) -> None:
        super().__init__(device=device)

        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."

        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        
        assert t >= 0.0 and t <= 1.0, f"t must be in [0,1]. Got t={t}."

        return SchedulerOutput(
            alpha_t=(1 - t)**self.n,
            sigma_t=t**self.n,
            d_alpha_t=-self.n * ((1-t) ** (self.n - 1)),
            d_sigma_t=self.n * (t ** (self.n - 1)),
        )
    
    def snr_inverse(self, snr: Tensor) -> Tensor:
        t = 1 / (1+snr ** (1/self.n))
        return t


    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        assert 1==2, "Not implemented"
        return kappa
    


class VPScheduler(Scheduler):
    """Variance Preserving Scheduler."""
    # beta_start: float = 0.0001,
    # beta_end: float = 0.02, -> SD (T=1000 -> 0)
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, device=0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(device=device)

    def __call__(self, t: Tensor) -> SchedulerOutput:
        
        # assert torch.all(t >= 0.0).item() and torch.all(t <= 1.0), f"t must be in [0,1]. Got t={t}."

        b = self.beta_min
        B = self.beta_max
        # T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b # FM
        # dT = -(1 - t) * (B - b) - b

        T = 0.5 * t ** 2 * (B - b) + t * b # Flux
        dT = t * (B - b) + b

        return SchedulerOutput(
            alpha_t=torch.exp(-0.5 * T),
            sigma_t=torch.sqrt(1 - torch.exp(-T)),
            d_alpha_t=-0.5 * dT * torch.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T)),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        T = -torch.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        # t = 1 - ((-b + torch.sqrt(b**2 + 2 * (B - b) * T)) / (B - b)) # FM 
        t = (-b + torch.sqrt(b**2 + 2 * (B-b) * T)) / (B - b) # Flux
        return t


class LinearVPScheduler(Scheduler):
    """Linear Variance Preserving Scheduler."""
    def __init__(self, device=0) -> None:
        super().__init__(device=device)

    def __call__(self, t: Tensor) -> SchedulerOutput:
        
        assert t >= 0.0 and t <= 1.0, f"t must be in [0,1]. Got t={t}."

        return SchedulerOutput(
            # alpha_t=t,
            # sigma_t=(1 - t**2) ** 0.5,
            # d_alpha_t=torch.ones_like(t),
            # d_sigma_t=-t / (1 - t**2) ** 0.5,
            alpha_t=(1 - t**2) ** 0.5,
            sigma_t=t,
            d_alpha_t=-t / (1 - t**2) ** 0.5,
            d_sigma_t=torch.ones_like(t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        # return torch.sqrt(snr**2 / (1 + snr**2)) # FM
        return torch.sqrt(1 / (snr**2 + 1)) # Flux


class CosineScheduler(Scheduler):
    """Cosine Scheduler."""
    def __init__(self, device=0) -> None:
        super().__init__(device=device)

    def __call__(self, t: Tensor) -> SchedulerOutput:
        
        assert t >= 0.0 and t <= 1.0, f"t must be in [0,1]. Got t={t}."

        pi = torch.pi
        return SchedulerOutput(
            alpha_t=torch.cos(pi / 2 * t),
            sigma_t=torch.sin(pi / 2 * t),
            d_alpha_t=-pi / 2 * torch.sin(pi / 2 * t),
            d_sigma_t=pi / 2 * torch.cos(pi / 2 * t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return 2.0 * torch.atan(1/snr) / torch.pi
