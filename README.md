# Pre-trained Encoder Inference: Revealing Upstream Encoders In Downstream Machine Learning Services

This is the official repository for the preprint ["Pre-trained Encoder Inference: Revealing Upstream Encoders In Downstream Machine Learning Services"](https://www.arxiv.org/abs/2408.02814) by Shaopeng Fu, Xuexue Sun, Ke Qing, Tianhang Zheng, and Di Wang.

## News

- 05/2025: An update version of the paper was released on [arXiv](https://www.arxiv.org/abs/2408.02814).
- 08/2024: The paper was released on [arXiv](https://www.arxiv.org/abs/2408.02814v1).

## Installation

### Requirements

- Python 3.11
- CUDA 11.8
- PyTorch 2.4.0

### Build environment via Anaconda

Download and install [Anaconda3](https://www.anaconda.com/download). Then, run following commands:

```bash
# create & activate conda environment
conda create -n encoder-inference python=3.11
conda activate encoder-inference

# install packages
conda install pytorch=2.4.0 torchvision=0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade huggingface_hub==0.24.5 transformers==4.41.2 diffusers==0.28.2 timm==1.0.7 accelerate==0.32.0 datasets==2.20.0 scipy==1.14.0 bitsandbytes==0.43.1
```

### Build environment via Docker

The docker building file is [./Dockerfile](./Dockerfile). Run following commands, and then the built image is `encoder-inference:latest`.

```bash
docker pull pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime
docker build --tag 'encoder-inference' .
```

**PS:** If you plan to use Docker to run your experiments, don't forget to **mount your default cache folder (e.g., `${HOME}/.cache`) to `/root/.cache` in the Docker container**.

## Quick Start

Example scripts and configurations are collected in folders [./scripts](./scripts) and [./configs](./configs), respectively.

Tutorials of running different experiments are collected in folder [./tutorials](./tutorials). They are:

- **PEI Attack vs Image Classification Services:** [tutorials/exp-img.md](./tutorials/exp-img.md).
- **PEI-assisted Adversarial Attack vs LLaVA:** [tutorials/exp-llava.md](tutorials/exp-llava.md).

## Citation

```
@article{fu2024pre,
  title={Pre-trained Encoder Inference: Revealing Upstream Encoders In Downstream Machine Learning Services},
  author={Shaopeng Fu and Xuexue Sun and Ke Qing and Tianhang Zheng and Di Wang},
  journal={arXiv preprint arXiv:2408.02814v2},
  year={2024}
}
