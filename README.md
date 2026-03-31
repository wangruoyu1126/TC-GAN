# TC-GAN

PyTorch implementation of **"Independence Constrained Disentangled Representation Learning from Epistemological Perspective"** (arXiv:2409.02672).

## High-level Idea

Disentangled representation learning aims to learn latent variables that correspond to semantically meaningful generative factors.

Your paper argues that disentanglement can be understood as the combination of two objectives:

1. **Informativeness (mutual information)**: latent variables should be predictive of generated images (capture semantics).
2. **Independence (total correlation / independence constraint)**: latent variables should be independent (reduce entanglement).

Inspired by epistemology, the paper introduces a **two-level latent space**:

- **Atomic level**: factors that are mutually independent.
- **Complex level**: factors derived from atomic factors; these can be dependent.

This repository trains a GAN-style model with an auxiliary network (`Q`) for semantics (InfoGAN-style) and TC variants for independence (FactorVAE-style).

## Project Structure

- `config.py`: training hyperparameters and dataset selection (`params[...]`)
- `train.py`: InfoGAN-style baseline training
- `train_factorvae.py`: TC discriminator / TC-loss training (closer to the paper's independence constraint)
- `train_btcvae.py`: another TC / bottleneck training variant
- `mnist_generate.py`: MNIST qualitative visualization / latent traversal
- `dataloader.py`: dataset wrappers (MNIST, FashionMNIST, SVHN, CelebA, dSprites, ...)
- `models/`: network definitions (and the TC discriminator for the dSprites setup)

## Dependencies

Typical dependencies include:

- PyTorch + torchvision
- numpy
- matplotlib
- h5py (if you use datasets that rely on it)

## How to Run

### 1) Configure the experiment

Edit `config.py` and set:

- `params['dataset']` (e.g., `MNIST`, `FashionMNIST`, `dSprites`, ...)
- `params['num_epochs']`, `params['batch_size']`, and learning rates (`lr_g`, `lr_d`, `lr_tcd`)

### 2) Train

The current README workflow in this repo uses `--exp_name` to create an experiment output folder.

Train an InfoGAN-style baseline:

```sh
python3 train.py --exp_name your_experiment_folder
```

Train the TC / independence-constrained variant:

```sh
python3 train_factorvae.py --exp_name your_experiment_folder --info_weight 0.1 --tc_weight 0.001
```

Output folders:

- generated images are saved as `Epoch_<epoch> <dataset>` inside `--exp_name`
- checkpoints are saved as:
  - `model_epoch_<epoch>_<dataset>`
  - `model_final_<dataset>`

### 3) MNIST qualitative visualization

After training a MNIST checkpoint, run:

```sh
python3 mnist_generate.py --load_path /path/to/your_experiment_folder/model_epoch_<epoch>_MNIST
```

## Results

### Quantitative Evaluation (Table 1 in the paper; dSprites)

The paper reports mean and standard deviation over **10 runs** and highlights the best/second-best score per metric.

| Method | EXP | JEMMIG | MOD | SAP | Z-diff |
|---|---:|---:|---:|---:|---:|
| VAE | 0.42| 0.20 | 0.87 | 0.11 | 0.69 |
| beta-VAE | 0.49 | 0.26 | 0.82 | 0.16 | 0.86 |
| AnnealedVAE | 0.72 | 0.33 | 0.97 | 0.39 | 0.86 |
| Factor-VAE | 0.41  | 0.19 | 0.92 | 0.21 | 0.80 |
| beta-TCVAE | 0.68  | 0.12 | 0.90  | 0.22 | 0.87 |
| InfoGAN | 0.54  | 0.08 | 0.56 | 0.05 | 0.76 |
| IB-GAN | 0.78 | 0.02 | 0.86  | 0.19 | 0.84 |
| InfoGAN-CR | 0.62 | 0.38 | 0.95 | 0.41 | 0.99 |
| **TC-GAN (ours)** | **0.85** | **0.45** | **0.98** | **0.48** | **0.99** |


