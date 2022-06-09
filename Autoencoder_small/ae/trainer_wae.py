import numpy as np

import torch
import torch.nn.functional as F
from ae.distributions import rand_cirlce2d


# regularizer used in WAE-MMD
def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    """Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()

    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum


def maximum_mean_discrepancy(z_tilde,
                                distribution_fn=rand_cirlce2d,
                                device='cpu'):
    """ Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    batch_size = z_tilde.size(0)
    z = distribution_fn(batch_size).to(device)
    
    assert z_tilde.ndimension() == 2
    n = z.size(0)
    out = im_kernel_sum(z, z, 1, exclude_diag=True).div(n*(n-1)) + \
        im_kernel_sum(z_tilde, z_tilde, 1, exclude_diag=True).div(n*(n-1)) - \
        im_kernel_sum(z, z_tilde, 1, exclude_diag=False).div(n*n).mul(2)
    return out


class WAEBatchTrainer:
    """ Wasserstein Autoencoder Batch Trainer.

        Args:
            autoencoder (torch.nn.Module): module which implements autoencoder framework
            optimizer (torch.optim.Optimizer): torch optimizer
            distribution_fn (callable): callable to draw random samples
            p (int): power of distance metric
            weight (float): weight of divergence metric compared to reconstruction in loss
            device (torch.Device): torch device
    """
    def __init__(self, autoencoder, optimizer, distribution_fn,
                 p=2, weight=200.0, device=None):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_.encoder.embedding_dim_
        self.p_ = p
        self.weight = weight
        self._device = device if device else torch.device('cpu')

    def __call__(self, x):
        return self.eval_on_batch(x)

    def train_on_batch(self, x):
        self.optimizer.zero_grad()
        evals = self.eval_on_batch(x)
        evals['loss'].backward()
        self.optimizer.step()
        return evals

    def test_on_batch(self, x):
        self.optimizer.zero_grad()
        evals = self.eval_on_batch(x)
        return evals

    def eval_on_batch(self, x):
        x = x.to(self._device)
        recon_x, z = self.model_(x)
        bce = F.binary_cross_entropy(recon_x, x)
        l1 = F.l1_loss(recon_x, x)
        _swd = maximum_mean_discrepancy(z, self._distribution_fn,
                                           self._device)
        w2 = float(self.weight) * _swd  # approximate wasserstein-2 distance
        loss = bce + l1 + w2
        return {
            'loss': loss,
            'bce': bce,
            'l1': l1,
            'w2': w2,
            'encode': z,
            'decode': recon_x
        }
