<div align="center">

# LaST​<sub>0</sub>​: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

[🌐**Project Page**](https://vla-last0.github.io/) | [✍️**Paper(Arxiv)**](https://arxiv.org/abs/2601.05248v2) | [🎥**Demo**](https://vla-last0.github.io/)

Zhuoyang Liu, Jiaming Liu, Hao Chen, Jiale Yu, Ziyu Guo, Chengkai Hou, Chenyang Gu, Xiangju Mi, Renrui Zhang, Kun Wu, Zhengping Che, Jian Tang, Pheng-Ann Heng, Shanghang Zhang

</div>

![](asset/method.png)
**🤖LaST​<sub>0</sub>​ is a framework that enables efficient reasoning before acting through a Latent Spatio-Temporal Chain-of-Thought (CoT), capturing fine-grained physical and robotic dynamics that are often difficult to verbalize.** Specifically, we introduce a token-efficient latent CoT space that models future visual dynamics, 3D structural information, and robot proprioceptive states, and further extends these representations across time to enable temporally consistent implicit reasoning trajectories. Furthermore, LaST<sub>0</sub> adopts a dual-system architecture implemented via a Mixture-of-Transformers design, where a reasoning expert conducts low-frequency latent inference and an acting expert generates high-frequency actions conditioned on robotics-oriented latent representations. To facilitate coordination, LaST<sub>0</sub> is trained with heterogeneous operation frequencies, enabling adaptive switching during deployment.

## ✨ News ✨

- [2026/03/05] The code of LaST​<sub>0</sub> is released! Including training and evaluating on various benchmarks. We also release our pretrained and finetuned checkpoints! The real-world scripts is comming soon! 🚀
- [2026/02/02] A new version of LaST<sub>0</sub> has been updated on arXiv, with more real-world experiments added, including tabletop manipulation, mobile manipulation, and dexterous hand manipulation! 🚀
- [2026/01/08] LaST​<sub>0</sub> is now live on arXiv! The code is also comming soon. 🚀

## 📦 Installation

The code is built using Python 3.10, we also recommand to use Python above Python 3.10. We require PyTorch >= 2.2.0 and CUDA >= 12.0 (It may run with lower versions, but we have not tested it).
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and create an environment as follows:

```bash
cd last0
conda create -n last0 python=3.10
conda activate last0

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install Flash Attention 2
# =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install "flash-attn==2.5.5" --no-build-isolation
```

If you want to test on LIBERO or RLBench, refer to the corresponding section to view environmental details.


## 🧩 Framework

Our code is built based on [Janus](https://github.com/deepseek-ai/Janus) and [Mirage](https://github.com/UMass-Embodied-AGI/Mirage) and is organized in the following framework:

- `conf`: config files for training
- `experiments`: launch scripts for LIBERO evaluation
- `scripts`: scripts for training and RLBench evaluation
- `janus`: contains last0 models, including last0 backbone & flow-matching & vlm & vla
- `transformers`: modified version of transformers, adding the MoT architecture
- `util`: contains different kinds of tools funtion
- `vla`: from openvla's vla structure, including action tokenizer, etc.

## 🤗 Model Zoo

We release our pretrained model's parameters on [hugging face](https://huggingface.co/) as follows:

- Robotic Large-Scale Pretrained Checkpoint for Action Expert: [Action Chunk = 8](https://huggingface.co/miniFranka/LaST0_Pretrain_AE_chunk8), [Action Chunk = 16](https://huggingface.co/miniFranka/LaST0_Pretrain_AE_chunk16)
- LIBERO SFT Checkpoints: [LIBERO Spatial](https://huggingface.co/miniFranka/LaST0_SFT_LIBERO_spatial), [LIBERO Object](https://huggingface.co/miniFranka/LaST0_SFT_LIBERO_object), [LIBERO Goal](https://huggingface.co/miniFranka/LaST0_SFT_LIBERO_goal), [LIBERO 10](https://huggingface.co/miniFranka/LaST0_SFT_LIBERO_10)
- [RLBench SFT Checkpoint](https://huggingface.co/miniFranka/LaST0_SFT_RLBench)


## 💡Getting Started

For quick evaluation, download the released checkpoints and test on these scripts:

```bash
# LIBERO (action_dim=7; action_chunk=8 for LIBERO_Spatial, action_chunk=16 for other 3 settings)
bash experiments/test_libero.sh

# RLBench (action_dim=7, action_chunk=1)
cd scripts
bash test_rlbench.sh
```

## 💾 Data Construction

We provide the processed LIBERO data in `.npy` format on [libero data](https://huggingface.co/datasets/miniFranka/libero_npy).

Constructing the latent CoT data is very important for LaST​<sub>0</sub>, and we provide the preprocess scripts:

```bash
cd utils

# for LIBERO
python gen_libero_json_stats.py

# for RLBench
python gen_rlbench_json_stats.py
```

The RLBench data includes point cloud, and LIBERO not. You can refer to these two scripts to build your own datasets.



## 📈 Fully Fine-Tuning

To fully fine-tune the pretrained models, we use `accelerate` package.

First, download our pretrain model for action expert, and change ``PRETRAIN_ACTION_PATH`` to your local ckpt absolute path.

Then launch the training script. We use one node with 8 A100 GPUs as an example.

```bash
cd scripts
bash train.sh
```

For the LIBERO benchmark and some datasets without point cloud data, we provide a clean version without point cloud. And the latent size is reduced to `8`.

```bash
cd scripts
bash train_wopc.sh
```

## 🔍Test in LIBERO

![](asset/libero.png)

We evaluated our LaST​<sub>0</sub>​ in [LIBERO](https://libero-project.github.io/main.html) and get the state-of-the-art performance.
First, install the LIBERO denpendencies:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

For more details of the eval scripts, refer to [OpenVLA-OFT](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md).
Then run the evaluation script:

```bash
# /path/to/last0
bash experiments/test_libero.sh
```

## 🔍Test in RLBench

We also evaluated our hybridvla in [RLBench](https://github.com/stepjam/RLBench), which based on the CoppeliaSim simulator. Install the virtual environment for testing in RLBench according to the following steps and begin your test. Thanks to the amazing work [LIFT3D](https://github.com/PKU-HMI-Lab/LIFT3D).

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

cd LIFT3D/third_party/RLBench
pip install -e .
cd ../../..

cd LIFT3D
pip install -e .
cd ..
```



## 📜️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 BibTeX

```tex
@misc{liu2026last0latentspatiotemporalchainofthought,
      title={LaST$_{0}$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model}, 
      author={Zhuoyang Liu and Jiaming Liu and Hao Chen and Jiale Yu and Ziyu Guo and Chengkai Hou and Chenyang Gu and Xiangju Mi and Renrui Zhang and Kun Wu and Zhengping Che and Jian Tang and Pheng-Ann Heng and Shanghang Zhang},
      year={2026},
      eprint={2601.05248},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.05248}, 
}
```
