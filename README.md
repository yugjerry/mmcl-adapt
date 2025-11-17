# Intrinsic Dimension Adaptation in Multi-Modal Contrastive Learning

This repository contains the code for reproducing the experiments in the paper:

**[Intrinsic Dimension and Temperature in Contrastive Learning](https://arxiv.org/abs/2505.12473)**  

## Overview

Multi-modal contrastive learning as a self-supervised representation learning technique has achieved great success in foundation model training, such as CLIP. In this paper, we study the theoretical properties of the learned representations from multi-modal contrastive learning beyond linear representations and specific data distributions. Our analysis reveals that, enabled by temperature optimization, multi-modal contrastive learning not only maximizes mutual information between modalities but also adapts to intrinsic dimensions of data, which can be much lower than user-specified dimensions for representation vectors. Experiments on both synthetic and real-world datasets demonstrate the ability of contrastive learning to learn low-dimensional and informative representations, bridging theoretical insights and practical performance.

## Repository Structure

```
├── utils.py              # Neural network architectures and data generation
├── _train_norm.py       # Main training script for contrastive learning
├── process_utils.py     # Utility functions for post-processing results
├── process.py           # Script for aggregating and visualizing results
├── data/                # Data directory (to be created)
│   ├── yfcc/
│   ├── imagenetv2/
│   └── citeseq/
└── README.md
```

## Usage

### Training Models

#### Synthetic Data
```bash
python3 -u _train_norm.py --dataset 'synthetic' --repN 50 --epoch_num 800 --n 10000 --n_append 2000 --d_z 5 --outdim 20
```

#### YFCC Dataset
```bash
python3 -u _train_norm.py --dataset 'yfcc' --repN 50 --epoch_num 600 --n 8000 --joint True --n_append 1000 --outdim 20
```

#### ImageNet-v2 Dataset
```bash
python3 -u _train_norm.py --dataset 'imagenetv2' --repN 50 --epoch_num 800 --n 8000 --joint True --n_append 1000 --outdim 20
```

#### CITE-seq Dataset
```bash
python3 -u _train_norm.py --dataset 'citeseq' --epoch_num 400 --n 10000 --n_append 8000 --repN 50 --outdim 20
```

### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset: `synthetic`, `yfcc`, `imagenetv2`, `citeseq` | `synthetic` |
| `--n` | Training sample size | `10000` |
| `--n_append` | Additional samples for normalization | `1000` |
| `--outdim` | Output (projection) dimension index (1-15) | `20` |
| `--arch` | Architecture: `linear`, `nonlinear`, `deep` | `deep` |
| `--link` | Link function (synthetic only): `linear`, `nonlinear` | `linear` |
| `--joint` | Use joint embeddings (vision datasets) | `False` |
| `--tau_tune` | Enable temperature tuning | `True` |
| `--epoch_num` | Number of training epochs | `800` |
| `--batch_size` | Batch size | `256` |
| `--lr` | Learning rate | `1e-4` |
| `--wd` | Weight decay | `1e-4` |
| `--repN` | Number of independent runs | `2` |

**Note:** The `outdim` parameter is an index (1-15) that maps to actual dimensions via:
```python
dim_seq = np.int64(np.linspace(1, 44, 15))
actual_dim = dim_seq[outdim - 1]
```

### Processing and Visualization

After training multiple models with different output dimensions:

```bash
# Process results and generate plots
python process.py --dataset 'synthetic' --arch deep --link linear --repN 50

# For other datasets
python process.py --dataset 'yfcc' --arch deep --repN 50 --joint True
python process.py --dataset 'imagenetv2' --arch deep --repN 50 --joint False
python process.py --dataset 'citeseq' --arch deep --repN 50
```


### Output Files

Training creates the following structure:
```
./{dataset}/file_{d_z}_{arch}_{link}_{repN}/
    file_{d_z}_{arch}_{link}_{repN}_{outdim}/
        rep_{i_rp}_{n}_{d_x}_{d_y}_{d_z}_{dataset}_{outdim}_{joint}.npz
        acc_{i_rp}_{n}_{d_x}_{d_y}_{d_z}_{dataset}_{outdim}_{joint}.npz
```

Each `.npz` file contains:
- Learned representations (`X_rep`, `Y_rep`, `X_rep_test`, `Y_rep_test`)
- Training metrics (`loss_clip`, `loss_align`, `tau_seq`)
- Downstream task accuracies
- Label information


## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{gui2025multi,
  title={Multi-modal contrastive learning adapts to intrinsic dimensions of shared latent variables},
  author={Gui, Yu and Ma, Cong and Ma, Zongming},
  journal={arXiv preprint arXiv:2505.12473},
  year={2025}
}
```


## Acknowledgments

This work builds upon:
- [CLIP](https://github.com/openai/CLIP) for vision-language pretraining
- [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning) for contrastive losses
- [scikit-dimension](https://github.com/j-bac/scikit-dimension) for intrinsic dimension estimation
