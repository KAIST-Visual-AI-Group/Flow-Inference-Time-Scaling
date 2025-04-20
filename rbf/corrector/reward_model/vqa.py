import os 
import torch
from dataclasses import dataclass
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

# import t2v_metrics
# import ImageReward as RM
from rbf import shared_modules as sm
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.image_utils import image_grid
from .vqa_server import RemoteVQAManager

class VQARewardModel:
    @ignore_kwargs
    @dataclass
    class Config:
        batch_size: int
        n_particles: int
        text_prompt: str

        image_reward_weight: float = 0.0
        vqa_server_addr: int = 5000

        ### DEBUGGER
        benchmark: bool = False
        img_idx: int = 0

    def __init__(self, cfg):
        self.cfg = self.Config(**cfg)
        
        RemoteVQAManager.register("process_VQA");
        sam_manager = RemoteVQAManager(address=("localhost", self.cfg.vqa_server_addr), authkey=b"secret");
        sam_manager.connect();
        self.vqa_function = sam_manager.process_VQA;
        
        self.toPIL = transforms.ToPILImage()

    
    def preprocess(self, images):
        pil_image_list = list()

        for idx in range(images.shape[0]):
            pil_image = self.toPIL(images[idx].to(torch.float32).cpu())
            pil_image_list.append(pil_image)
        
        return pil_image_list
    
    def __call__(self, images, step):

        output = self.vqa_function({
            "images": images,
            "text": self.cfg.text_prompt,
            "irw": self.cfg.image_reward_weight,
        });

        score = output.get("scores");
        score = torch.tensor(score);
        
        if not sm.DO_NOT_SAVE_INTERMEDIATE_IMAGES:
            self.logging(step, images, score)
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
