import os 
import torch
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont
import re 

from typing import Optional, Dict, List
from dataclasses import dataclass

import numpy as np
from transformers import CLIPProcessor, AutoModel
import torchvision

from rbf import shared_modules as sm
from rbf.utils.image_utils import torch_to_pil, image_grid
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_info, print_error, print_qna, print_warning


class PickScoreRewardModel:
    @ignore_kwargs
    @dataclass
    class Config:
        text_prompt: str = None
        batch_size: int = 1
        n_particles: int = 1

        disable_debug: bool = False
        log_interval: int = 5

        device: int = 0


    def __init__(
        self, 
        cfg,
    ):
        
        self.cfg = self.Config(**cfg)

        self.disable_debug = self.cfg.disable_debug
        self.log_interval = self.cfg.log_interval
        self.device = self.cfg.device 

        self.reward_logs = []
        self.image_logs = []

        # processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        # self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        # self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(self.cfg.device)

        self.dtype = sm.prior.pipeline.dtype
        self.device = self.cfg.device 

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        checkpoint_path = "yuvalkirstain/PickScore_v1"
        # checkpoint_path = f"{os.path.expanduser('~')}/.cache/PickScore_v1"
        self.model = AutoModel.from_pretrained(checkpoint_path).eval().to(self.cfg.device, dtype=self.dtype)

        self.target_size =  224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])


    def preprocess(self, images):
        return images 
    

    def __call__(
        self, 
        images: torch.Tensor, 
        step: int,
    ):

        print_info("Reward model using text prompt:", self.cfg.text_prompt) if not sm.OFF_LOG else None

        text_inputs = self.processor(
            text=self.cfg.text_prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)

        inputs = torchvision.transforms.Resize(self.target_size)(images)
        inputs = self.normalize(inputs).to(self.dtype)

        image_embeds = self.model.get_image_features(pixel_values=inputs)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)

        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image)

        rewards = scores.item()

        self.reward_logs.append(rewards)
        self.image_logs.append(torch_to_pil(images))

        if len(self.reward_logs) % (self.cfg.batch_size * self.cfg.n_particles) == 0:
            if not self.cfg.disable_debug:
                self.log_preds(step)

        return scores
        

    def log_preds(self, step):
        if not self.cfg.disable_debug and step % self.log_interval == 0:
            str_rewards = " | ".join([f"{r:.2f}" for r in self.reward_logs])
            if self.image_logs:
                _n = 1 if len(self.image_logs) % 2 == 1 else 2
                grid = image_grid(
                    self.image_logs,
                    _n, len(self.image_logs) // _n,
                )

                try:
                    font = ImageFont.truetype("arial.ttf", 80) 
                except IOError:
                    font = ImageFont.load_default() 
                
                draw = ImageDraw.Draw(grid)
                draw.text((0, 0), str_rewards, (255,0,0),font=font)

                grid.save(
                    os.path.join(sm.logger.debug_dir, f"pickscore_{step}.png")
                )

        print_info(f"Reward logs step {step}: {self.reward_logs}") if not sm.OFF_LOG else None
        
        self.image_logs = []
        self.reward_logs = []
        
        
    