# TC-GAN

PyTorch implementation of [Independence Constrained Disentangled Representation Learning from Epistemological Perspective](https://arxiv.org/pdf/2409.02672?).

## Introduction



```


## Usage
Edit the **`config.py`** file to select training parameters and the dataset to use. Choose *`dataset`* from **['MNIST', 'FashionMNIST', 'SVHN', 'CelebA']**

To train the model run **`train.py`**:
```sh
python3 train.py
```
After training the network to experiment with the latent code for the `MNIST` dataset run **`mnist_generate.py`**:
```sh
python3 mnist_generate.py --load_path /path/to/pth/checkpoint
```

## Results
### MNIST
<!-- <table align='center'>
<tr align='center'>
<th> Training Data </th>
<th> Generation GIF </th>
</tr>
<tr>
<td><img src = 'results/mnist_results/Training Images MNIST.png' height = '450'>
<td><img src = 'results/mnist_results/infoGAN_MNIST.gif' height = '450'>
</tr>
</table>

<table align='center'>
<tr align='center'>
<th> Epoch 1 </th>
<th> Epoch 50 </th>
<th> Epoch 100 </th>
</tr>
<tr>
<td><img src = 'results/mnist_results/Epoch_1_MNIST.png' height = '300'>
<td><img src = 'results/mnist_results/Epoch_50_MNIST.png' height = '300'>
<td><img src = 'results/mnist_results/Epoch_100_MNIST.png' height = '300'>
</tr>
</table> -->


