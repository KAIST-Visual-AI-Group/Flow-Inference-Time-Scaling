import os 
import torch
from PIL import Image
from abc import ABC, abstractmethod
from PIL import Image 
import re 
from PIL import Image, ImageDraw, ImageFont

from typing import Optional, Dict, List
from dataclasses import dataclass
import ImageReward as RM

import numpy as np
from transformers import AutoProcessor, AutoModel

from rbf import shared_modules as sm
from rbf.utils.image_utils import torch_to_pil, image_grid
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_info, print_error, print_qna, print_warning


class ImageRewardRewardModel:
    @ignore_kwargs
    @dataclass
    class Config:
        text_prompt: str = None
        batch_size: int = 1
        n_particles: int = 1
        device: int = 0

        reward_scale: float = 1.0

        disable_debug: bool = False
        log_interval: int = 5
        


    def __init__(
        self, 
        cfg,
    ):
        
        self.cfg = self.Config(**cfg)

        self.disable_debug = self.cfg.disable_debug
        self.log_interval = self.cfg.log_interval

        self.reward_logs = []
        self.image_logs = []

        self.model = RM.load("ImageReward-v1.0")


    def preprocess(self, images):
        return torch_to_pil(images)

    
    def __call__(
        self,
        image: Image,
        step: int,
    ):

        rewards = self.model.score(self.cfg.text_prompt, [image])

        self.reward_logs.append(rewards)
        self.image_logs.append(image)

        if len(self.image_logs) == (self.cfg.batch_size * self.cfg.n_particles):
            self.log_preds(step)

        return self.cfg.reward_scale * torch.tensor(rewards)
        
        

    def log_preds(self, step):
        if not self.cfg.disable_debug and step % self.log_interval == 0:
            if self.image_logs:
                resized_images = []
                ori_w, ori_h = self.image_logs[0].size
                if (ori_w > 768) or (ori_h > 768):
                    ori_w, ori_h = ori_w // 2, ori_h // 2
                
                for idx in range(len(self.image_logs)):
                    resized_images.append(self.image_logs[idx].resize((ori_w, ori_h)))

                grid_width = ori_w * self.cfg.n_particles
                grid_height = ori_h * self.cfg.batch_size

                grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

                try:
                    font = ImageFont.truetype("arial.ttf", 40) 
                except IOError:
                    font = ImageFont.load_default() 

                draw = ImageDraw.Draw(grid_image)

                try:
                    for i in range(self.cfg.batch_size):
                        for j in range(self.cfg.n_particles):
                            idx = i * self.cfg.n_particles + j
                            grid_image.paste(resized_images[idx], (j * 256, i * 256))
                            draw.text((j * 256 + 10, i * 256 + 10), f"{self.reward_logs[idx]:.5f}", font=font, fill=(255, 0, 0))

                except Exception as e:
                    grid_height = 256
                    grid_width = 256 * len(resized_images)
                    grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

                    # Sequential row
                    for i in range(len(resized_images)):
                        grid_image.paste(resized_images[i], (i * 256, 0))
                        draw.text((i * 256 + 10, 10), f"{self.reward_logs[i]:.5f}", font=font, fill=(255, 0, 0))

                grid_path = os.path.join(sm.logger.debug_dir, f"{step:03d}.png")
                grid_image.save(grid_path)


        print_info(f"Reward logs step {step}: {self.reward_logs}") if not sm.OFF_LOG else None
        
        self.image_logs = []
        self.reward_logs = []
        
        