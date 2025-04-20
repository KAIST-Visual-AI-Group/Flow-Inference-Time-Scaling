from multiprocessing.managers import BaseManager
import torch
import argparse
import time
import t2v_metrics
import ImageReward as RM

class RemoteVQAManager(BaseManager):
    pass

@torch.no_grad()
def process_VQA(input):
    start_time = time.time();

    images = input.get("images");
    text_prompt = input.get("text");
    image_reward_weight = input.get("irw");
    
    scores = []
    for idx in range(0, len(images), vqa_batch_size):
        cur_batch_size = min(vqa_batch_size, len(images) - idx)
        cur_images = images[idx:idx+cur_batch_size]
        cur_scores = vqa_reward_model(images=cur_images, texts=[text_prompt])

        if image_reward_weight > 0.0:
            image_reward_score = rm_model.score(text_prompt, cur_images)
            if type(image_reward_score) is float:
                image_reward_score = [image_reward_score] 
            image_reward_score = torch.tensor(image_reward_score).to(cur_scores)
            cur_scores = cur_scores + image_reward_score[:, None] * image_reward_weight

        scores += cur_scores.reshape(-1).cpu().numpy().tolist();

    output = {
        "scores": scores
    };

    print("Reward calculation took {:.3f}s".format(time.time() - start_time));
    return output;


if __name__ == "__main__":
    parser = argparse.ArgumentParser();

    parser.add_argument("--gpu", type = str, default = "0");
    parser.add_argument("--addr", type = int, default = 5000);
    parser.add_argument("--vqa_model", type = str, default = "clip-flant5-xxl");
    parser.add_argument("--vqa_batch_size", type = int, default = 32);
    
    args = parser.parse_args();

    RemoteVQAManager.register("process_VQA", callable = process_VQA);

    device = "cuda:{}".format(args.gpu);
    vqa_batch_size = args.vqa_batch_size;
    vqa_model = args.vqa_model;

    vqa_reward_model = t2v_metrics.get_score_model(model = vqa_model, device = device);
    rm_model = RM.load("ImageReward-v1.0", device = device);

    manager = RemoteVQAManager(address=("localhost", int(args.addr)), authkey=b"secret")
    server = manager.get_server()
    print("Server started... Listening for requests.")
    server.serve_forever()
