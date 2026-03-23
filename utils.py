import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchvision import transforms
import h5py

from torch.utils.data import Dataset
import os
DIR = os.path.abspath(os.path.dirname(__file__))

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean()) # + 90 * var.square().mean()
        # print('nll loss', nll)
        # print('var_reg', var.square().mean())
        return nll

def noise_sample(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)

    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(dis_c_dim, size=batch_size)
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx











########################
# For factorVAE method
########################
def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


def reparametrize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + std * eps


########################
# For betaTCVAE method
########################
def get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    # print('latent_sample， latent_dist', latent_sample.shape, latent_dist[0].shape, latent_dist[1].shape)
    # print('mat_log_qz', mat_log_qz.shape)
    # print('mat_log_qz.sum(2)', mat_log_qz.sum(2).shape)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    # print('log_qz', log_qz.shape)
    #
    # print('torch.logsumexp(mat_log_qz, dim=1, keepdim=False)', torch.logsumexp(mat_log_qz, dim=1, keepdim=False).shape)
    # print('log_prod_qzi', log_prod_qzi.shape)
    # assert log_qz == 0

    return log_pz, log_qz, log_prod_qzi, log_q_zCx





def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    # print('normalization', normalization.shape)

    inv_var = torch.exp(-logvar)
    # print('inv_var', inv_var.shape)

    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    # print('log_density', log_density.shape)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

















class ColoredMNIST(datasets.VisionDataset):
    def __init__(self, root='data/cmnist', train=True, transform=transforms.ToTensor(), target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        if train == True:
            self.data_label_tuples = torch.load(root + '/train.pt')
        else:
            self.data_label_tuples = torch.load(root + '/test.pt')

    def __len__(self):
        return len(self.data_label_tuples)

    def __getitem__(self, index):
        img, target, red = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, red





class doubleMNIST(datasets.VisionDataset):
    def __init__(self, root='data/double_mnist', train=True, transform=transforms.ToTensor(), target_transform=None):
        super(doubleMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        if train == True:
            self.data_label_tuples = torch.load(root + '/train.pt')
        else:
            self.data_label_tuples = torch.load(root + '/test.pt')

    def __len__(self):
        return len(self.data_label_tuples)

    def __getitem__(self, index):
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



class dSprites(Dataset):
    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "dsprite_train.npz"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    background_color = 0
    lat_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195,
                                           0.64442926, 0.80553658, 0.96664389, 1.12775121,
                                           1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                           2.57771705, 2.73882436, 2.89993168, 3.061039,
                                           3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                           4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                           5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799, 6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  'color': np.array([1.])}

    def __init__(self, root=os.path.join(DIR, 'data/dsprites/')):
        super().__init__()

        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose([transforms.ToTensor()])

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]
        return sample, lat_value








class dshapes(datasets.VisionDataset):
    def __init__(self, root='data/3dshapes', train=True, transform=transforms.ToTensor(), target_transform=None):
        super(dshapes, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        if train == True:
            dataset = h5py.File('3dshapes.h5', 'r')
            self.data_label_tuples = torch.load(root + '/train.pt')


    def __len__(self):
        return len(self.data_label_tuples)

    def __getitem__(self, index):
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target







