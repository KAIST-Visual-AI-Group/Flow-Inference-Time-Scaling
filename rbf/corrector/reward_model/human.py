import os 
import torch
from abc import ABC, abstractmethod
from PIL import Image 
import re 

from typing import Optional, Dict, List
from dataclasses import dataclass

import numpy as np

from rbf import shared_modules as sm
from rbf.utils.image_utils import torch_to_pil, image_grid
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_info, print_error, print_qna, print_warning


class HumanRewardModel:
    @ignore_kwargs
    @dataclass
    class Config:
        text_prompt: str = None
        batch_size: int = 10

        disable_debug: bool = False
        log_interval: int = 5

        reward_type: str = ""
        criterion: str = "" # granularity
        class_names: str = "" # counting 
        class_gt_counts: str = "" # counting
        reward_func: str = "exp" # counting
        reward_scale: float = 1.0 # counting
        sharpness: float = 1.0 # counting
        concept_erasure: str = "" # concept erasure

    def __init__(
        self, 
        cfg,
    ):
        
        self.cfg = self.Config(**cfg)

        self.disable_debug = self.cfg.disable_debug
        self.log_interval = self.cfg.log_interval

        self.reward_logs = []
        self.image_logs = []

        if self.cfg.reward_type == "granularity":
            print_info(f"{self.cfg.criterion} is the criterion")
            assert self.cfg.criterion, "Criterion must be provided for granularity reward type"
            assert self.cfg.criterion != "", "Criterion must be provided for granularity reward type"

        elif self.cfg.reward_type == "counting":
            print_info(f"Task: {self.cfg.reward_type}. Detection information is {self.cfg.class_gt_counts} and {self.cfg.class_names}")
            self.class_gt_counts = [int(n) for n in self.cfg.class_gt_counts.split(",")]
            self.class_names = [[t.strip()] for t in self.cfg.class_names.split(",")] + [[" "]]

        elif self.cfg.reward_type == "erasure":
            self.concept_erasure = self.cfg.concept_erasure

        else:
            raise ValueError(f"Invalid reward type {self.cfg.reward_type}")


    def preprocess(self, images):
        return torch_to_pil(images)

    
    def __call__(
        self,
        image: List,
        step: int,
    ):

        _n = 1 if len(image) % 2 == 1 else 2
        grid_image = image_grid(
            image,
            _n, len(image) // _n,
        )
        
        img_save_path = os.path.join(sm.logger.debug_dir, f"human_feedback.png")
        grid_image.save(img_save_path)

        if self.cfg.reward_type == "granularity":
            human_reward = f"See image at {img_save_path} and select the most preferred image that best captures the text prompt {self.cfg.criterion}. Enter the number of correct markings of the criterion: "

        elif self.cfg.reward_type == "counting":
            human_reward = input(f"See image at {img_save_path} and count the objects specified in the order: {self.class_names}. Enter the counts separated by commas: ")
            human_cnt = [int(n) for n in human_reward.split(",")]
            diff = np.sum((np.array(self.class_gt_counts) - np.array(human_cnt)) ** 2) # L2 
            if self.cfg.reward_func == "exp":
                reward = self.cfg.reward_scale * np.exp(-self.cfg.sharpness * diff) 
            elif self.cfg.reward_func =="inverse_l1":
                reward = self.cfg.reward_scale * 1 / (1 + diff)
            elif self.cfg.reward_func == "inverse":
                reward = self.cfg.reward_func * 1 / (diff + 0.1)
            else:
                raise NotImplementedError(f"Unknown reward function: {self.cfg.reward_func}")

        elif self.cfg.reward_type == "erasure":
            human_reward = input(f"See image at {img_save_path} and select the image that does not display the specified concepts: {self.concept_erasure}. Enter the number of correct erasures for each image separated by comma: ")
            reward = [int(_r) / len(self.concept_erasure.split(",")) for _r in human_reward.split(",")]
        
        elif self.cfg.reward_type == "ethics":
            human_reward = input(f"See image at {img_save_path} and select the image that does not display {self.concept_erasure}")
            reward = float(reward)

        else:
            raise ValueError(f"Invalid reward type {self.cfg.reward_type}")
        

        self.reward_logs = reward
        self.image_log = grid_image

        self.log_preds(step)

        return torch.tensor([reward])
        
        

    def log_preds(self, step):
        if not self.disable_debug and (step % self.log_interval == 0):
            self.image_log.save(
                os.path.join(sm.logger.debug_dir, f"human_{step}.png")
            )

        print_info(f"Reward logs step {step}: {self.reward_logs}")
        self.reward_logs = []
        
        