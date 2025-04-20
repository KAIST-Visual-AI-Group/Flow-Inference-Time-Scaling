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

### Compositional Image Generation
You can host the VQAScore VLM on a separate device to save GPU memory. By default, the server responds on port 5000:
```
python rbf/corrector/reward_model/vqa_server.py
```

