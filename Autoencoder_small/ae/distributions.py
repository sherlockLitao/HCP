import numpy as np

import torch
from sklearn.datasets import make_circles


def rand_cirlce2d(batch_size):
   
    r = np.random.uniform(size=(batch_size))
    theta = 2 * np.pi * np.random.uniform(size=(batch_size))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.array([x, y]).T
    return torch.from_numpy(z).type(torch.FloatTensor)


def rand_ring2d(batch_size):
    
    circles = make_circles(2 * batch_size, noise=.01)
    z = np.squeeze(circles[0][np.argwhere(circles[1] == 0), :])
    return torch.from_numpy(z).type(torch.FloatTensor)


def rand_uniform2d(batch_size):
    
    z = 2 * (np.random.uniform(size=(batch_size, 2)) - 0.5)
    return torch.from_numpy(z).type(torch.FloatTensor)


def rand(dim_size):
    def _rand(batch_size):
        return torch.rand((batch_size, dim_size))
    return _rand


def randn(dim_size):
    def _randn(batch_size):
        return torch.randn((batch_size, dim_size))
    return _randn
