<h1 align="center">Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing</h1>

<!-- Title image -->
<p align="center">
  <img src="./asset/teaser.jpg" width="600"/>
</p>

<!-- Badges -->
<p align="center">
  <a href="https://arxiv.org/abs/2503.19385">
    <img src="https://img.shields.io/badge/arXiv-2503.19385-b31b1b.svg" />
  </a>
  <a href="https://flow-inference-time-scaling.github.io/">
    <img src="https://img.shields.io/badge/Website-rbf.github.io-blue.svg" />
  </a>
</p>

<!-- Authors -->
<p align="center">
  <a href="https://jh27kim.github.io">Jaihoon Kim*</a>,
  <a href="https://github.com/taehoon-yoon">Taehoon Yoon*</a>,
  <a href="https://github.com/Jisung0111">Jisung Hwang*</a>,
  <a href="https://mhsung.github.io">Minhyuk Sung</a>
</p>

<blockquote align="center">
  Our inference-time scaling precisely aligns pretrained flow models with user preferences—such as text prompts, object quantities, and more.
</blockquote>


<!-- Release Note -->
### Release
- **[21/04/25]** 🔥 We have released the implementation of _Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing_ for compositional image generation. Code for quantity-aware and aesthetic image generation will be released soon.


<!-- Setup -->
### Setup

Create and activate a Conda environment (tested with Python 3.10):

```
conda create -n rbf python=3.10
conda activate rbf
```

Clone the repository:
```
git clone https://github.com/KAIST-Visual-AI-Group/Flow-Inference-Time-Scaling.git
cd Flow-Inference-Time-Scaling
```

Install PyTorch (tested with version 2.1.0) and required dependencies:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
pip install -e .
```

For compositional image generation, we use VQAScore, which can be installed with the following command:
```
cd third-party/t2v_metrics/
pip install -e .
```

### Configuration:
  - `--text_prompt` : Text prompt to guide generation. Required for reward-based sampling.

  - `--filtering_method` : Strategy to select or prune particles (`bon`, `smc`, `code`, `svdd`, `rbf`).

  - `--batch_size` : Number of prompts or samples processed in parallel during inference.

  - `--n_particles` : Number of particles used per timestep to explore the reward landscape.

  - `--block_size` : Number of timesteps grouped together for blockwise updates (set to 1 except in `code.yaml`).

  - `--convert_scheduler` : Apply interpolant conversion at inference time (`vp`).

  - `--sample_method` : Sampling method (`sde`, `ode`).

  - `--diffusion_norm` : SDE sampling diffusion norm.

  - `--max_nfe` : Total computational budget (in number of function evaluations) available during sampling. 

  - `--max_steps` : Number of denoising steps in the generative process.

  - `--reward_score` : Reward function used for alignment (`vqa`, `counting` ,`aesthetic`).

  - `--init_n_particles` : Initial number of particles at the start of generation.


</details>

### Compositional Image Generation
Host the VQAScore VLM on a separate device to save GPU memory. By default, the server responds on port 5000:
```
python rbf/corrector/reward_model/vqa_server.py
```

Run compositional image generation using the following command. To prevent out-of-memory (OOM) issues, we recommend running it on a different device from the VQA server. 

You may optionally override configuration values by specifying arguments directly in the command line:
```
CUDA_VISIBLE_DEVICES={$DEVICE} python main.py --config config/compositional_image/rbf.yaml text_prompt={$TEXT_PROMPT}
```


### Quantity-Aware Image Generation
Coming Soon.


### Aesthetic Image Generation
Coming Soon.


## Citation
If you find our code helpful, please consider citing our work:
```
@article{kim2025inference,
  title={Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing},
  author={Kim, Jaihoon and Yoon, Taehoon and Hwang, Jisung and Sung, Minhyuk},
  journal={arXiv preprint arXiv:2503.19385},
  year={2025}
}
```

<br />

## Acknowledgement 
This repository incorporates implementations from [Flow Matching](https://github.com/facebookresearch/flow_matching) and [FLUX](https://github.com/black-forest-labs/flux). We sincerely thank the authors for publicly releasing their codebases. 
