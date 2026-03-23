import torch
# from utils import *
import math


q_mu = torch.load('q_mu.pt')
q_var = torch.load('q_var.pt')
latent_sample = torch.load('latent_sample.pt')
# log_qz = torch.load('log_qz.pt')
# log_prod_qzi = torch.load('log_prod_qzi.pt')

print('q_mu', q_mu.shape, q_mu[55:65])
print('q_var', q_var.shape, q_var[55:65])


def get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape
    # print('batch_size, hidden_dim', batch_size, hidden_dim)

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    assert 1 == 0

    # print('log_q_zCx', log_q_zCx) - have nan

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

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
    # print('x\n', x)
    # print('mu\n', mu)
    # print('logvar\n', logvar)

    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    # print('normalization\n', normalization)

    inv_var = torch.exp(-logvar)
    # print('inv_var\n', inv_var)

    # print('(x - mu)\n', (x - mu))
    # print('(x - mu)**2\n', ((x - mu)**2))
    # print('(x - mu)**2 * inv_var\n', ((x - mu)**2 * inv_var))

    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    print('log_density\n', log_density)

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








torch.set_printoptions(profile="full")
latent_dist = (q_mu, q_var)
_, log_qz, log_prod_qzi, _ = get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                       latent_dist,
                                                       n_data=737280,
                                                       is_mss=True)

# print('log_qz', log_qz)
# print('log_prod_qzi', log_prod_qzi)
