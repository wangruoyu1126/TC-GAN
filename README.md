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
| VAE [19] | 0.42 ± .00 | 0.20 ± .00 | 0.87 ± .01 | 0.11 ± .00 | 0.69 ± .03 |
| beta-VAE [14] | 0.49 ± .00 | 0.26 ± .00 | 0.82 ± .01 | 0.16 ± .00 | 0.86 ± .01 |
| AnnealedVAE [3] | 0.72 ± .01 | 0.33 ± .00 | _0.97 ± .00_ | 0.39 ± .01 | 0.86 ± .05 |
| Factor-VAE [17] | 0.41 ± .00 | 0.19 ± .00 | 0.92 ± .01 | 0.21 ± .00 | 0.80 ± .02 |
| beta-TCVAE [7] | 0.68 ± .01 | 0.12 ± .00 | 0.90 ± .00 | 0.22 ± .00 | 0.87 ± .03 |
| InfoGAN [8] | 0.54 ± .00 | 0.08 ± .00 | 0.56 ± .02 | 0.05 ± .00 | 0.76 ± .04 |
| IB-GAN [16] | _0.78 ± .02_ | 0.02 ± .01 | 0.86 ± .03 | 0.19 ± .01 | 0.84 ± .04 |
| InfoGAN-CR [22] | 0.62 ± .00 | _0.38 ± .00_ | 0.95 ± .00 | _0.41 ± .00_ | _0.99 ± .02_ |
| **TC-GAN (ours)** | **0.85 ± .01** | **0.45 ± .00** | **0.98 ± .00** | **0.48 ± .00** | **0.99 ± .01** |

### Qualitative Results (Latent Traversal Figures)

Insert qualitative figure demonstrations here (as in the paper). Replace the `TODO` links with the paths to your generated images.

| Dataset | Example Traversal | Figure Placeholder |
|---|---|---|
| MNIST | Traverse on rotation while keeping digit fixed | `![MNIST traversal TODO](./figures/mnist_traversal.png)` |
| FashionMNIST | Traverse on thickness while keeping item class fixed | `![FashionMNIST traversal TODO](./figures/fashion_traversal.png)` |
| dSprites | Traverse one factor while keeping others fixed (e.g., rotation vs shape/position) | `![dSprites traversal TODO](./figures/dsprites_traversal.png)` |

## Citation

If you use this project in your work, please cite:

`Independence Constrained Disentangled Representation Learning from Epistemological Perspective (arXiv:2409.02672).`
