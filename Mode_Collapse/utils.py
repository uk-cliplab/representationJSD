from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np


def get_generator(latent_size: int) -> nn.Module:
    """
    Returns the generator network.
    :param latent_size: (int) Size of the latent input vector
    :return: (nn.Module) Simple feed forward neural network with three layers,
    """
    return nn.Sequential(nn.Linear(latent_size, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.Tanh(),
                         nn.Linear(256, 2, bias=True))


# def get_generator(latent_size: int) -> nn.Module:
#     """
#     Returns the generator network.
#     :param latent_size: (int) Size of the latent input vector
#     :return: (nn.Module) Simple feed forward neural network with three layers,
#     """
#     return nn.Sequential(nn.Linear(latent_size, 128, bias=True),
#                          nn.ReLU(),
#                          nn.Linear(128, 128, bias=True),
#                          nn.ReLU(),
#                          nn.Linear(128, 2, bias=True))


# def get_discriminator(use_spectral_norm: bool) -> nn.Module:
#     """
#     Returns the generator network.
#     :param latent_size: (int) Size of the latent input vector
#     :return: (nn.Module) Simple feed forward neural network with three layers,
#     """
#     return nn.Sequential(nn.Linear(2, 128, bias=True),
#                          nn.ReLU(),
#                          nn.Linear(128, 128, bias=True),
#                          nn.ReLU(),
#                          nn.Linear(128, 2, bias=True))


def get_discriminator(use_spectral_norm: bool) -> nn.Module:
    """
    Returns the discriminator network.
    :param use_spectral_norm: (bool) If true spectral norm is utilized
    :return: (nn.Module) Simple feed forward neural network with three layers and probability output.
    """
    if use_spectral_norm:
        return nn.Sequential(spectral_norm(nn.Linear(2, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 1, bias=True)))
    return nn.Sequential(nn.Linear(2, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 1, bias=True))


class RFF_tuneable(torch.nn.Module):
    def __init__(self, d_latent, d_out,sigma_init,freezeRFF=False, seed = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        w_init = torch.randn([d_latent,d_out], dtype = torch.float )
        bias_init = 2*np.pi*torch.rand(d_out, dtype=torch.float)

        if freezeRFF:
            self.weight = torch.nn.Parameter(w_init,requires_grad = False)
            self.bias = torch.nn.Parameter(bias_init,requires_grad = False)
        else:
            self.weight = torch.nn.Parameter(w_init,requires_grad = True)
            self.bias = torch.nn.Parameter(bias_init,requires_grad = True)

        self.sigma = torch.nn.Parameter(sigma_init, requires_grad=False)
        self.d_out = d_out # necessary to normalize the FF
        

    def forward(self, x):
        w_times_x= torch.matmul(x, (1/self.sigma)*self.weight)
        return np.sqrt(1/self.d_out)*torch.cat((torch.cos(w_times_x), torch.sin(w_times_x)), dim=-1)


def get_DFFN(use_spectral_norm: bool) -> nn.Module:
    """
    Returns the discriminator network.
    :param use_spectral_norm: (bool) If true spectral norm is utilized
    :return: (nn.Module) Simple feed forward neural network with three layers and probability output.
    """
    if use_spectral_norm:
        return nn.Sequential(spectral_norm(nn.Linear(2, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 256, bias=True)),
                             nn.LeakyReLU(),
                             spectral_norm(nn.Linear(256, 256, bias=True)),
                             nn.LeakyReLU(),
                             RFF_tuneable(256,8, torch.tensor(1.0)))
    return nn.Sequential(nn.Linear(2, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         RFF_tuneable(256,8, torch.tensor(1.0)))

# def get_DFFN(use_spectral_norm: bool) -> nn.Module:
#     """
#     Returns the discriminator network.
#     :param use_spectral_norm: (bool) If true spectral norm is utilized
#     :return: (nn.Module) Simple feed forward neural network with three layers and probability output.
#     """
#     if use_spectral_norm:
#         return nn.Sequential(spectral_norm(nn.Linear(2, 256, bias=True)),
#                              nn.LeakyReLU(),
#                              spectral_norm(nn.Linear(256, 256, bias=True)),
#                              nn.LeakyReLU(),
#                              spectral_norm(nn.Linear(256, 256, bias=True)),
#                              nn.LeakyReLU(),
#                              spectral_norm(nn.Linear(256, 256, bias=True)),
#                              nn.LeakyReLU(),
#                              RFF_tuneable(256,256, torch.tensor(1.0)))
#     return nn.Sequential(nn.Linear(2, 128, bias=True),
#                          nn.ReLU(),
#                          nn.Linear(128, 128, bias=True),
#                          nn.ReLU(),
#                          RFF_tuneable(128,128, torch.tensor(1.0)))


def get_data(samples: Optional[int] = 400, variance: Optional[float] = 0.05) -> torch.Tensor:
    """
    Function generates a 2d ring of 8 Gaussians
    :param samples: (Optional[int]) Number of samples including in the resulting dataset. Must be a multiple of 8.
    :param variance: (Optional[float]) Variance of the gaussian
    :return: (torch.Tensor) generated data
    """
    assert samples % 8 == 0 and samples > 0, "Number of samples must be a multiple of 8 and bigger than 0"
    # Init angels of the means
    angels = torch.cumsum((2 * np.pi / 8) * torch.ones((8)), dim=0)
    # Convert angles to 2D coordinates
    means = torch.stack([torch.cos(angels), torch.sin(angels)], dim=0)
    # Generate data
    data = torch.empty((2, samples))
    counter = 0
    for gaussian in range(means.shape[1]):
        for sample in range(int(samples / 8)):
            data[:, counter] = torch.normal(means[:, gaussian], variance)
            counter += 1
    # Reshape data
    data = data.T
    # Shuffle data
    data = data[torch.randperm(data.shape[0])]
    # Convert numpy array to tensor
    return data.float()
