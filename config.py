# Dictionary storing network parameters.
params = {
    'batch_size': 64,
    'num_epochs': 100,
    'lr_g': 0.001,
    'lr_d': 0.002,
    'lr_tcd': 0.001,
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch': 1,
    'dataset': 'FashionMNIST', # MNIST, CMNIST, SVHN, CelebA, FashionMNIST, dSprites, doubleMNIST, tripleMNIST
    'label_smoothing': True}