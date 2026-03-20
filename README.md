# EISAM 2025
### EISAM: Leveraging Extragradient for Effective Sharpness-Aware Minimization in Deep Learning

The following figure illustrates the optimization trajectories of EISAM, SAM, and SGD over 100 steps on a 3D toy loss landscape.

<p align="center">
  <img src="figures/3d-100step.gif" width="720" alt="Optimization trajectories of EISAM, SAM, and SGD on a 3D toy loss landscape">
</p>

**Figure 1.** Optimization paths of different optimizers (EISAM vs. SAM vs. SGD) on a synthetic 3D loss surface after 100 steps.

---

## Table of Contents
- [EISAM 2025](#eisam-2025)
  - [Table of Contents](#table-of-contents)
  - [Environments](#environments)
  - [Requirements](#requirements)
    - [Image Classification](#image-classification)
    - [Object\_detectio](#object_detectio)
  - [Quick Start – Image Classification](#quick-start--image-classification)
  - [Object Detection](#object-detection)
  - [Dataset Preparation](#dataset-preparation)
  - [Contact](#contact)
  - [Acknowledgments](#acknowledgments)

---

## Environments

**Image Classification**
- OS: Ubuntu 22.04
- GPU: GeForce RTX 3090
- Python: 3.10.12
- PyTorch: 2.8.0+cu128
- CUDA: 12.8
- Driver: 570.172.08

**Object Detection**
- OS: Ubuntu 24.04
- GPU: GeForce RTX 5090
- PyTorch: 2.9.1+cu130
- CUDA: 13.0
- Driver: 581.57

---

## Requirements

### Image Classification
create a new conda environment with the configuration file image.yml
```bash
conda env create -f image.yml
conda activate eisam
```

### Object_detectio
create a new venv environment with the configuration file requirements.txt
```bash
python3 -m venv venv_eisam
source venv_eisam/bin/activate
pip install -r requirements.txt
```

---

## Quick Start – Image Classification

Here are ready-to-run examples on CIFAR-100 (ResNet-50):

**1. Vanilla SGD (with Nesterov Acceleration)**
```bash
python main.py --seed 1 --data_name CIFAR-100 --num_classes 100 --arch resnet50 --optimizer SGD --learning_rate 0.05 --batch_size_train 128 --weight_decay 5.0e-4 --gpu_id 0 --epochs 200 --project test --scheduler cosine --use_cutmix True
```

**2. Sharpness-Aware Minimization (SAM)**
```bash
python main.py --seed 1 --data_name CIFAR-100 --num_classes 100 --arch resnet50 --optimizer SGD --learning_rate 0.05 --batch_size_train 128 --weight_decay 5.0e-4 --gpu_id 0 --epochs 200 --project test --scheduler cosine --use_sam 1 --sam_rho 0.05 --use_cutmix True
```

**3. EISAM (Ours)**
```bash
python main.py --seed 1 --data_name CIFAR-100 --num_classes 100 --arch resnet50 --optimizer SGD --learning_rate 0.05 --batch_size_train 128 --weight_decay 1.0e-3 --gpu_id 0 --epochs 200 --project test --scheduler cosine --use_eisam 1 --eisam_rho 0.05 --eisam_s 0.001 --use_cutmix True
```

---

## Object Detection
For each optimizer, we provide a dedicated Bash script (`train_XXX.sh`) that includes all optimizer-specific hyperparameters and common training configuration parameters. These scripts document the default settings used in our experiments.

When training on the COCO dataset (`--dataset coco`), we set the batch size to 5 and the total number of epochs to 50 (different from the LVIS setting). Models are trained from scratch on COCO, whereas LVIS experiments initialize with COCO-pretrained weights.

Note: The pairing between datasets and models is hard-coded in the `od_main.py` file. To switch to a different model, please manually modify this file.

**Example: Training with EISAM**
```bash
bash train_eisam.sh
```

---

## Dataset Preparation
Since the datasets are relatively large, we only provide the directory structure and download links here:

**Directory Structure**
```bash
data
└── coco2017
    ├── annotations/
    ├── test2017/
    ├── train2017/
    └── val2017/
```

**Download Links**

- **COCO 2017** (images + annotations): [Official Download Page](https://cocodataset.org/#download)

- **LVIS v1.0** (annotations only): LVIS shares the same image set as COCO 2017. Download the following files and place `lvis_v1_train.json` and `lvis_v1_val.json` into the `data/coco2017/annotations/` directory:

  - [lvis_v1_train.json.zip](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip)
  - [lvis_v1_val.json.zip](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip)

After setting the `--dataset coco` (or `--dataset lvis`) flag and the correct `--dataset_root`, `od_main.py` will automatically match and load the corresponding annotation files.

**Pre-trained Weights**: When training on LVIS for the first time, `od_main.py` will automatically download the pre-trained model weights.

---

## Contact
If you have any questions about the implementation or experiments, please feel free to contact me:
  - Email: fyao56@stu.xjtu.edu.cn

---

## Acknowledgments
The image classification codebase is adapted from SALA: A Novel Optimizer for Accelerating
**Convergence and Improving Generalization:**
https://github.com/cltan023/sala2023
