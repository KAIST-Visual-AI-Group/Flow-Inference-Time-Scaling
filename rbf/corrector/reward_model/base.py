import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import randint
import contextlib
import io

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize, Normalize, Compose
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as transforms
import torchvision

from rbf import shared_modules as sm
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_warning, print_error, print_info


ASSETS_PATH = os.path.join(os.path.realpath(__file__), "aesthetic_score")


# class MLPDiff(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(768, 1024),
#             nn.Dropout(0.2),
#             nn.Linear(1024, 128),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.Dropout(0.1),
#             nn.Linear(64, 16),
#             nn.Linear(16, 1),
#         )


#     def forward(self, embed):
#         return self.layers(embed)
    
#     def forward_up_to_second_last(self, embed):
#         # Process the input through all layers except the last one
#         for layer in list(self.layers)[:-1]:
#             embed = layer(embed)
#         return embed


# class AestheticRewardModel(torch.nn.Module):
#     @ignore_kwargs
#     @dataclass
#     class Config:
#         ckpt_root: str = ""
#         ckpt: str = "sac+logos+ava1-l14-linearMSE.pth"
#         n_particles: int = 1
#         batch_size: int = 1
#         benchmark: bool = False
#         img_idx: int = 0

#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = self.Config(**cfg)

#         ckpt_path = os.path.join(self.cfg.ckpt_root, self.cfg.ckpt)

#         self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#         self.mlp = MLPDiff()

#         # print("Loading model from ", ckpt_path)
#         state_dict = torch.load(ckpt_path, weights_only=True)
#         self.mlp.load_state_dict(state_dict)

#         self.preprocessor = Compose([
#             Resize(224, antialias=False),
#             Normalize(
#                 mean=[0.48145466, 0.4578275, 0.40821073],
#                 std=[0.26862954, 0.26130258, 0.27577711])
#         ])
#         self.toPIL = transforms.ToPILImage()


#     def __call__(self, images, step, decoded_tweedies_list=None):
#         embed = self.clip.get_image_features(pixel_values=images)
#         embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
#         score = self.mlp(embed).squeeze(1)

#         if not sm.DO_NOT_SAVE_INTERMEDIATE_IMAGES and decoded_tweedies_list is not None:
#             img_list = list()
#             for idx in range(decoded_tweedies_list.shape[0]):
#                 img = self.toPIL(decoded_tweedies_list[idx].to(torch.float32).cpu())
#                 img_list.append(img)
#             self.logging(step, img_list, score)
#         return score
    
#     def logging(self, step, images, score):
#         resized_images = list()
#         img_size_ = 512
#         for idx in range(len(images)):
#             resized_images.append(images[idx].resize((img_size_, img_size_)))

#         grid_width = img_size_ * self.cfg.n_particles
#         grid_height = img_size_ * self.cfg.batch_size

#         grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

#         try:
#             font = ImageFont.truetype("arial.ttf", 80) 
#         except IOError:
#             font = ImageFont.load_default() 

#         draw = ImageDraw.Draw(grid_image)

#         try:
#             for i in range(self.cfg.batch_size):
#                 for j in range(self.cfg.n_particles):
#                     idx = i * self.cfg.n_particles + j
#                     grid_image.paste(resized_images[idx], (j * img_size_, i * img_size_))
#                     draw.text((j * img_size_ + 10, i * img_size_ + 10), f"{score[idx]:.5f}", font=font, fill=(255, 0, 0))

#         except Exception as e:
#             grid_height = img_size_
#             grid_width = img_size_ * len(resized_images)
#             grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

#             # Sequential row
#             for i in range(len(resized_images)):
#                 grid_image.paste(resized_images[i], (i * img_size_, 0))
#                 draw.text((i * img_size_ + 10, 10), f"{score[i]:.5f}", font=font, fill=(255, 0, 0))


#         if self.cfg.benchmark:
#             os.makedirs(os.path.join(sm.logger.debug_dir, f"{self.cfg.img_idx:05d}"), exist_ok=True)
#             grid_path = os.path.join(os.path.join(sm.logger.debug_dir, f"{self.cfg.img_idx:05d}"), f"{step:03d}.png")
#         else:
#             grid_path = os.path.join(sm.logger.debug_dir, f"{step:03d}.png")
#         grid_image.save(grid_path)

#     def preprocess(self, images):
#         _dtype = images.dtype
#         return self.preprocessor(images).to(_dtype)
    

#     def generate_feats(self, images):
#         embed = self.clip.get_image_features(pixel_values=images)
#         embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
#         return embed
    

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class AestheticRewardModel(nn.Module):
    @ignore_kwargs
    @dataclass
    class Config:
        ckpt_root: str = ""
        ckpt: str = "sac+logos+ava1-l14-linearMSE.pth"
        n_particles: int = 1
        batch_size: int = 1
        benchmark: bool = False
        img_idx: int = 0

    def __init__(self, cfg, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device, dtype=self.dtype)
        self.mlp = MLP().to(self.device, dtype=self.dtype)

        state_dict = torch.load(os.path.join(self.cfg.ckpt_root, "sac+logos+ava1-l14-linearMSE.pth"), map_location=self.device)
        self.mlp.load_state_dict(state_dict)

        self.target_size =  224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                          std=[0.26862954, 0.26130258, 0.27577711])
        self.toPIL = transforms.ToPILImage()
        
        self.eval()

    # def __call__(self, images):
    #     inputs = torchvision.transforms.Resize(self.target_size)(images)
    #     inputs = self.normalize(inputs).to(self.dtype)
    #     embed = self.clip.get_image_features(pixel_values=inputs)
    #     embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

    #     return self.mlp(embed).squeeze(1)

    def __call__(self, images, step, decoded_tweedies_list=None):
        # embed = self.clip.get_image_features(pixel_values=images)
        # embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

        inputs = torchvision.transforms.Resize(self.target_size)(images)
        inputs = self.normalize(inputs).to(self.dtype)
        embed = self.clip.get_image_features(pixel_values=inputs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        score = self.mlp(embed).squeeze(1)

        if not sm.DO_NOT_SAVE_INTERMEDIATE_IMAGES and decoded_tweedies_list is not None:
            img_list = list()
            for idx in range(decoded_tweedies_list.shape[0]):
                img = self.toPIL(decoded_tweedies_list[idx].to(torch.float32).cpu())
                img_list.append(img)
            self.logging(step, img_list, score)
        return score
    
    def logging(self, step, images, score):
        resized_images = list()
        img_size_ = 512
        for idx in range(len(images)):
            resized_images.append(images[idx].resize((img_size_, img_size_)))

        grid_width = img_size_ * self.cfg.n_particles
        grid_height = img_size_ * self.cfg.batch_size

        grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

        try:
            font = ImageFont.truetype("arial.ttf", 80) 
        except IOError:
            font = ImageFont.load_default() 

        draw = ImageDraw.Draw(grid_image)

        try:
            for i in range(self.cfg.batch_size):
                for j in range(self.cfg.n_particles):
                    idx = i * self.cfg.n_particles + j
                    grid_image.paste(resized_images[idx], (j * img_size_, i * img_size_))
                    draw.text((j * img_size_ + 10, i * img_size_ + 10), f"{score[idx]:.5f}", font=font, fill=(255, 0, 0))

        except Exception as e:
            grid_height = img_size_
            grid_width = img_size_ * len(resized_images)
            grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

            # Sequential row
            for i in range(len(resized_images)):
                grid_image.paste(resized_images[i], (i * img_size_, 0))
                draw.text((i * img_size_ + 10, 10), f"{score[i]:.5f}", font=font, fill=(255, 0, 0))


        if self.cfg.benchmark:
            os.makedirs(os.path.join(sm.logger.debug_dir, f"{self.cfg.img_idx:05d}"), exist_ok=True)
            grid_path = os.path.join(os.path.join(sm.logger.debug_dir, f"{self.cfg.img_idx:05d}"), f"{step:03d}.png")
        else:
            grid_path = os.path.join(sm.logger.debug_dir, f"{step:03d}.png")
        grid_image.save(grid_path)

    def preprocess(self, images):
        return images

class CompressionRewardModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        

    def jpeg_compressibility(self, images):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        pil_images = [Image.fromarray(image) for image in images]

        sizes = []
        with contextlib.ExitStack() as stack:
            buffers = [stack.enter_context(io.BytesIO()) for _ in pil_images]
            for image, buffer in zip(pil_images, buffers):
                image.save(buffer, format="JPEG", quality=95)
                # sizes.append(buffer.tell() / 1000)  # Size in kilobytes
                sizes.append(buffer.tell() / (1000 * 1000))  # Size in megabytes
        
        return -np.array(sizes)
        # return 1 / (np.array(sizes) + 1e-8)


    def __call__(self, images, step):
        jpeg_compressibility_scores = self.jpeg_compressibility(images)
        return jpeg_compressibility_scores


    def preprocess(self, images):
        x0_image = images.cpu().permute(0, 2, 3, 1).float().numpy()
        target = (x0_image * 255).round().astype("uint8")

        return target


class InpaintingRewardModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()


    def __call__(self, images, step):
        mask = sm.model.gt_rgb_mask.unsqueeze(0)
        return 1. / (F.mse_loss(images * mask, sm.model.gt_rgb_image * mask) + 1e-8)
    
    def preprocess(self, images):
        return images