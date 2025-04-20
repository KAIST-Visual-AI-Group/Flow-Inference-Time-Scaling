from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import randint

import torch
import torch.nn.functional as F

from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_warning, print_error, print_info
from rbf import shared_modules as sm


class Corrector(ABC):
    @ignore_kwargs
    @dataclass
    class Config:
        correct_steps: int = 1


    def __init__(self, cfg):
        self.cfg = self.Config(**cfg)
        self.potentials = []


    @abstractmethod
    def pre_correct(self, images):
        # Correct samples
        pass 


    @abstractmethod
    def post_correct(self, images):
        # Correct samples
        pass 


class DDIMCorrector(Corrector):
    def pre_correct(
        self, 
        noisy_sample, 
        tweedie, 
        model_pred, 
        step=None
    ):
        return noisy_sample, model_pred

    def post_correct(
        self, 
        prev_noisy_sample, 
        tweedie,
        model_pred,
        step, 
    ):
        return prev_noisy_sample, tweedie, model_pred

    def final_correct(
        self, 
        noisy_sample, 
        tweedie, 
        step=None
    ):
        return noisy_sample, tweedie
    
