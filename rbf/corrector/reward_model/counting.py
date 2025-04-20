import os 
from PIL import Image 
from dataclasses import dataclass

import numpy as np
import torch
import supervision as sv

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration
from rbf import shared_modules as sm
from rbf.utils.image_utils import torch_to_pil, image_grid
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_info


class CountingRewardModel:
    @ignore_kwargs
    @dataclass
    class Config:
        device : int = 0
        batch_size: int = 10
        count_reward_model: str = "gdsam"
        disable_debug: bool = False
        log_interval: int = 5

        class_names: str = "airplane, camel"
        class_gt_counts: str = "3, 6"
        reward_func: str = "diff"


    def __init__(
        self, 
        cfg,
    ):
        
        self.cfg = self.Config(**cfg)

        print_info(f"Counting Reward Model Config: {self.cfg.count_reward_model}")

        if self.cfg.count_reward_model == "gdsam":
            bboxmaker_id = "IDEA-Research/grounding-dino-base";
            segmenter_id = "facebook/sam-vit-base";

            self.gd_processor = AutoProcessor.from_pretrained(bboxmaker_id);
            self.bboxmaker = AutoModelForZeroShotObjectDetection.from_pretrained(bboxmaker_id).to("cuda");
            self.sam_processor = AutoProcessor.from_pretrained(segmenter_id);
            self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to("cuda");

            # RemoteGDSAMManager.register("process_GDSAM");
            # sam_manager = RemoteGDSAMManager(address=("localhost", self.cfg.gdsam_server_addr), authkey=b"secret");
            # sam_manager.connect();
            # self.gdsam_function = sam_manager.process_GDSAM;

            self.bounding_box_annotator = sv.BoxAnnotator();
            self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER);

            # class_gt_counts: str = "n1, n2, ..., nN" where n1 is the number of instances of class 1
            self.class_gt_counts = torch.tensor([int(n) for n in str(self.cfg.class_gt_counts).split(",")]);
            self.class_names = [t.strip() for t in self.cfg.class_names.split(",")];
            self.class_texts = ". ".join(self.class_names) + ".";

            print_info(f"Class texts: {self.class_texts}")
            print_info(f"Class names: {self.class_names}")
            print_info(f"Class gt counts: {self.class_gt_counts}")
            
        else:
            raise NotImplementedError(f"Unknown reward model: {self.cfg.count_reward_model}")


        self.disable_debug = self.cfg.disable_debug
        self.log_interval = self.cfg.log_interval
        self.detection_logs = []
        self.reward_logs = []


    def preprocess(self, images):
        return torch_to_pil(images)


    def gdsam_reward(
        self,
        image: Image, 
        step: int
    ):
        texts = self.class_texts
        names = texts[:-1].split(". ")

        gd_inputs = self.gd_processor(images=image, text=texts, return_tensors="pt").to("cuda");
        outputs = self.bboxmaker(**gd_inputs)

        results = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            gd_inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]]
        )[0]

        del results["text_labels"]

        gt_labels = {label: i for i, label in enumerate(names)}
        results["labels"] = torch.tensor([gt_labels.get(label, -1) for label in results["labels"]])
        indices = torch.argwhere(results["labels"] >= 0).reshape(-1)
        
        for key in results:
            results[key] = results[key][indices].cpu();
        
        cnt = 0;
        indices = [];
        unq_labels = torch.unique(results["labels"]);
        for i in unq_labels:
            idx0 = torch.argwhere(results["labels"] == i).reshape(-1);
            scores = results["scores"][idx0];
            max_score = torch.max(scores).item();
            idx1 = torch.argwhere((scores > max_score * 0.6) | (scores > 0.32)).reshape(-1);
            indices.append(idx0[idx1]);
            cnt += 1;
        
        counts = [0 for _ in names];

        if cnt:
            indices = torch.cat(indices, dim = 0);
            for key in results: results[key] = results[key][indices];
        
            boxes = [results["boxes"].numpy().tolist()];

            sam_inputs = self.sam_processor(images = image, input_boxes = boxes, return_tensors = "pt").to("cuda");

            outputs = self.segmentator(**sam_inputs);
            masks = self.sam_processor.post_process_masks(
                masks = outputs.pred_masks,
                original_sizes = sam_inputs.original_sizes,
                reshaped_input_sizes = sam_inputs.reshaped_input_sizes
            )[0];
            masks = masks.float().permute(0, 2, 3, 1).mean(dim = -1) > 0;

            results["masks"] = masks;
            results["mask_sums"] = torch.sum(masks, dim = [1, 2]).cpu();
        
            indices0 = [];
            scores = torch.zeros(results["masks"][0].shape, dtype = torch.float32, device = "cuda");

            for label_i in unq_labels:
                idx0 = torch.argwhere(results["labels"] == label_i).reshape(-1);
                cuml_mask = torch.zeros_like(results["masks"][0]);

                for i0 in sorted(idx0.numpy().tolist(), key = lambda i: results["mask_sums"][i].item()):
                    ms = results["masks"][i0];
                    msum = results["mask_sums"][i0];
                    mpos = torch.sum(torch.bitwise_and(~cuml_mask, ms)).item();

                    if mpos / msum > 0.5:
                        indices0.append(i0);
                        cuml_mask |= ms;
                        scores[ms] = torch.maximum(scores[ms], torch.full_like(scores[ms], results["scores"][i0].item()));

            indices1 = [];
            for i0 in indices0:
                ms = results["masks"][i0];
                ssum = torch.sum(scores[ms] > results["scores"][i0].item()) / results["mask_sums"][i0].item();
                if ssum < 0.5:
                    indices1.append(i0);
                    counts[results["labels"][i0].item()] += 1;

            indices1 = torch.tensor(indices1, dtype = torch.int64);
            for key in results: results[key] = results[key][indices1];

        output = {
            "count": counts,
            "boxes": results["boxes"].numpy().tolist(),
            "label": results["labels"].numpy().tolist(),
            "score": results["scores"].numpy().tolist()
        };

        counts = output.get("count");

        if self.cfg.reward_func == "diff":
            diff = torch.sum((self.class_gt_counts - torch.tensor(counts)) ** 2);
            reward = -diff;
            
        else:
            raise NotImplementedError(f"Unknown reward function: {self.cfg.reward_func}")

        self.reward_logs.append(reward)
        
        if not self.disable_debug and (step % self.log_interval == 0):
            svimage = np.array(image);

            if sum(counts):
                detections = sv.Detections(
                    xyxy=np.array(output.get("boxes")),
                    class_id=np.array(output.get("label")),
                    confidence=np.array(output.get("score"))
                );

                labels = [
                    f"{class_id} {confidence:0.2f}"
                    for class_id, confidence
                    in zip(detections.class_id, detections.confidence)
                ];

                svimage = self.bounding_box_annotator.annotate(svimage, detections);
                svimage = self.label_annotator.annotate(svimage, detections, labels);

            self.detection_logs.append(
                Image.fromarray(svimage)
            );

        if len(self.detection_logs) == self.cfg.batch_size:
            self.log_preds(step);

        return reward


    def __call__(
        self,
        image: Image, 
        step: int,
    ):
        if self.cfg.count_reward_model == "gdsam":
            return self.gdsam_reward(image, step);
        
        else:
            raise NotImplementedError(f"Unknown reward model: {self.cfg.count_reward}")


    def log_preds(self, step):
        # NOTE: Particle-wise logging
        if self.detection_logs:
            _n = 1 if len(self.detection_logs) % 2 == 1 else 2
            grid = image_grid(
                self.detection_logs,
                _n, len(self.detection_logs) // _n,
            )

            grid.save(
                os.path.join(sm.logger.debug_dir, f"counting_{step}.png")
            )

        print_info(f"Reward logs step {step}: {self.reward_logs}")

        self.detection_logs = []
        self.reward_logs = []