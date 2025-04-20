import os 
import torch
from abc import ABC, abstractmethod
from PIL import Image 
import io
import re 

from typing import Optional, Dict
from dataclasses import dataclass

import numpy as np
import base64

from rbf import shared_modules as sm
from rbf.utils.image_utils import torch_to_pil, image_grid
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_info, print_error, print_qna, print_warning, print_note

    

class BaseLLMEngine(ABC):
    @abstractmethod
    def chat(self):
        pass
    
class OpenAILLMEngine(BaseLLMEngine):
    def __init__(
        self,
        text_prompt: str,
        reward_type: str,
        criterion: str,
        temperature: float = 0.3,
        iteration: int = 3,
        system_context: str = "You are a helpful assistant.",
        model_id: str = "gpt-4o-mini",
    ):
            
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        self.criterion = criterion
        self.temperature = temperature
        self.iteration = iteration
        self.system_context = system_context
        self.model_id = model_id

        print_info(f"Criterion given as: {self.criterion}")
        if reward_type == "granularity":
            self.query_prefix = f"Rate this image using the following criterion: {self.criterion} \
                1. For each description, mark whether the provided image suffices the description. So for each description the score will be either 1 (correct and present) or 0 (incorrect or absent). \
                2. Grade the image based on the summation of the marks obtained in Step 1. The final score will be the sum of all the marks obtained in Step 1."

            self.query_suffix = """
                Provide your answer using the following format:
                1. Description 1 - Justification 1. Your Marking.
                2. Description 2 - Justification 2. Your Marking.
                3. ...
                Score: #/N 
                Do not miss out any description. Provide the score at the end using the fraction format. For example, if the score is 5 out of 10, you should write 5/10.
            """

            self.vlm_question = f"{self.query_prefix}\n{text_prompt}\n{self.query_suffix}"
        
        elif reward_type == "composition":
            raise NotImplementedError("Composition reward type not implemented yet")

        else:
            raise NotImplementedError(f"Unknown reward type: {reward_type}")


        print(f"Using model engine {self.model_id} to generate text")


    def chat(
        self,
        step,
        base64_image: str,
        text_prompt: str = None,
    ):  
        if base64_image is not None:
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.vlm_question}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]
        
        else:
            assert text_prompt is not None, "Text prompt is required for text only question"
            base64_image = None

            messages=[
                {
                    "role": "user",
                    "content": f"{text_prompt}",
                }
            ]

        # Iterate for multiple times to get the reward
        for _i in range(self.iteration):
            response = self.client.chat.completions.create(
                model=self.model_engine,
                messages=messages,
                temperature=self.temperature,
            )

            output_text = response.choices[0].message.content

            if base64_image is None:
                return output_text
            
            match = re.search(r"Score:\s*(\d+/\d+)", output_text)

            if match:
                reward = eval(match.group(1))
                print_info("Reward VLM", f"Score: {reward}")
                break 
            
            else:
                reward = 0.5
                print_error(f"Reward not found at step {step}, iteration {_i+1}. Setting reward to 0.5")

        return output_text, reward
    



class QwenAPILLMEngine(BaseLLMEngine):
    def __init__(
        self,
        text_prompt: str,
        reward_type: str,
        criterion: str,
        temperature: float = 0.3,
        iteration: int = 3,
        system_context: str = "You are a helpful assistant.",
        model_id: str = "qwen2.5-vl-72b-instruct",
        concept_erasure: str = None,
    ):

        from openai import OpenAI

        self.client = OpenAI(
            #If the environment variable is not configured, please replace the following line with the Dashscope API Key: api_key="sk-xxx".
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

        self.text_prompt = text_prompt
        self.criterion = criterion
        self.temperature = temperature
        self.iteration = iteration
        self.system_context = system_context
        self.model_id = model_id

        self.concept_erasure = concept_erasure
        
        print_info(f"Criterion given as: {self.criterion}")
        if reward_type == "granularity":
            self.query_prefix = f"Rate this image using the following criterion: {self.criterion} \
                1. For each description, mark whether the provided image suffices the description. So for each description the score will be either 1 (correct and present) or 0 (incorrect or absent). \
                2. Grade the image based on the summation of the marks obtained in Step 1. The final score will be the sum of all the marks obtained in Step 1. \
                Text Prompt: "

            self.query_suffix = """
                Provide your answer using the following format:
                1. Description 1 - Justification 1. Your Marking.
                2. Description 2 - Justification 2. Your Marking.
                3. ...
                Score: #/N
                Do not miss out any description. Provide the score at the end using the fraction format. For example, if the score is 5 out of 10, you should write 5/10.
            """

            self.vlm_question = f"{self.query_prefix}\n{self.text_prompt}\n{self.query_suffix}"
        
        elif reward_type == "erasure":
            self.query_prefix = f"Your goal is to detect unwanted content that are present in the image. \
                Assess whether the image contains the following content: "

            self.query_suffix = """
                For each unwanted content, mark whether the provided image contains the content. So for each unwanted content the score will be either 1 (absent) or 0 (present). \
                Do not miss out any description. Provide the score at the end using the fraction format. For example, if the score is 5 out of 10, you should write 5/10.
            """

            self.vlm_question = f"{self.query_prefix}\n{self.concept_erasure}\n{self.query_suffix}"

        else:
            raise NotImplementedError(f"Unknown reward type: {reward_type}")

        print_note(f"VLM question: {self.vlm_question}")


    def chat(
        self,
        step,
        base64_image=None, 
        text_prompt=None, 
        min_pixels=512*28*28, 
        max_pixels=2048*28*28,
    ):
        
        if base64_image is not None:

            messages=[
                {
                    "role": "system",
                    "content": [{"type":"text","text": self.system_context}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {"type": "text", "text": self.vlm_question},
                    ],
                }
            ]

        else:
            assert text_prompt is not None, "Text prompt is required for text only question"
            assert base64_image is None, "Image is not required for text only question"

            base64_image = None
            messages=[
                {'role': 'system', 'content': self.system_context},
                {'role': 'user', 'content': text_prompt}
            ]
        

        # Iterate for multiple times to get the reward
        for _i in range(self.iteration):
            try:
                completion = self.client.chat.completions.create(
                    model = self.model_id,
                    messages = messages,
                    temperature=self.temperature,
                )
            except Exception as e:
                print_error(e)
                output_text = "Incomplete API call"
                reward = 0.5


            output_text = completion.choices[0].message.content # str

            if base64_image is None:
                return output_text
            
            # match = re.search(r"Score:\s*(\d+/\d+)", output_text)
            match = re.findall(r"\b\d+\/\d+\b", output_text)
            if match:
                reward = eval(match[0])
                print("Reward VLM", f"Score: {reward}")
                break 
        
            else:
                print_error(f"Reward not found at step {step}, iteration {_i+1}. Setting reward to 0.5")

        return output_text, reward
    



class VLMRewardModel:
    @ignore_kwargs
    @dataclass
    class Config:
        text_prompt: str = None
        batch_size: int = 10

        disable_debug: bool = False
        log_interval: int = 5

        vlm_type: str = "gpt"
        vlm_model: str = "gpt-4o-mini"

        reward_type: str = "granularity" 
        system_context: str = "You are a helpful assistant."

        vlm_iteration: int = 3

        criterion: str = ""
        concept_erasure: str = ""

        # Online model (API)
        temperature: float = 0.3

        # Offline model
        vlm_device: int = 1

    def __init__(
        self, 
        cfg,
    ):
        
        self.cfg = self.Config(**cfg)
        if self.cfg.vlm_type == "gpt":
            self.vlm = OpenAILLMEngine(
                text_prompt=self.cfg.text_prompt, 
                reward_type=self.cfg.reward_type,
                criterion=self.cfg.criterion,
                temperature=self.cfg.temperature,
                iteration=self.cfg.vlm_iteration,
                system_context=self.cfg.system_context,
                model_id=self.cfg.vlm_model,
            )
        
        elif self.cfg.vlm_type == "qwen_api":
            self.vlm = QwenAPILLMEngine(
                text_prompt=self.cfg.text_prompt, 
                reward_type=self.cfg.reward_type,
                criterion=self.cfg.criterion,
                temperature=self.cfg.temperature,
                iteration=self.cfg.vlm_iteration,
                system_context=self.cfg.system_context,
                model_id=self.cfg.vlm_model,
                concept_erasure=self.cfg.concept_erasure,
            )
        
        else:
            raise NotImplementedError(f"Unknown VLM model: {self.cfg.vlm_model}")


        self.disable_debug = self.cfg.disable_debug
        self.log_interval = self.cfg.log_interval

        self.reward_logs = []
        self.response_logs = []
        self.image_logs = []


    def preprocess(self, images):
        return images


    def __call__(
        self,
        image: torch.Tensor, 
        step: int,
    ):  
        import time 
        import random 

        time.sleep(random.random() * 2)
        
        pil_image = torch_to_pil(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")

        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        if self.cfg.vlm_type in ["gpt-4o-mini", "qwen_api"]:
            response_text, reward = self.vlm.chat(
                step=step,
                base64_image=base64_image,
            )

        else:
            raise NotImplementedError(f"Unknown VLM model: {self.cfg.vlm_model}")
        

        self.response_logs.append(response_text)
        self.reward_logs.append(reward)
        self.image_logs.append(pil_image)

        if len(self.reward_logs) == self.cfg.batch_size:
            self.log_preds(step)

        return torch.tensor(reward)


    def log_preds(self, step):
        # NOTE: Particle-wise logging
        if not self.disable_debug and step % self.log_interval == 0:
            _n = 2 if len(self.image_logs) % 2 == 0 else 1
            for _i in range(len(self.image_logs)):
                self.image_logs[_i] = self.image_logs[_i].resize((256, 256))

            grid = image_grid(self.image_logs, _n, len(self.image_logs) // _n)
            grid_path = os.path.join(sm.logger.debug_dir, f"vlm_images_{step}_{'{:.2f}'.format(self.reward_logs[0])}.png")
            grid.save(grid_path)

            response_log_path = os.path.join(sm.logger.debug_dir, f"vlm_response.txt")
            with open(response_log_path, "a") as f:
                f.write(f"============ Step {step} ============\n")
                for _i, _r in enumerate(self.response_logs):
                    f.write(f"Particle Index {_i+1} Particle Response: {_r} \n")
                    f.write("\n")
                f.write(f"\n Reward: {', '.join(list(map(str, self.reward_logs)))} \n")
                f.write("======================================\n")

        print_info(f"Reward logs step {step}: {self.reward_logs}")

        self.response_logs = []
        self.image_logs = []
        self.reward_logs = []
