from typing import Tuple
import torch

import utils

__INITIALIZER__ = {}


def register_initializer(name: str):
    def wrapper(func):
        if __INITIALIZER__.get(name, None):
            raise NameError(f"Name {name} is already registered.")
        __INITIALIZER__[name] = func
        return func
    return wrapper


def get_initializer(name: str):
    if __INITIALIZER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __INITIALIZER__[name]


@register_initializer(name='gaussian')
def gaussian_initializer(shape: Tuple[int]) -> torch.Tensor:
    return torch.rand(shape)


@register_initializer(name='spectral')
def spectral_initializer(amplitude: torch.Tensor) -> torch.Tensor:
    intensity = amplitude ** 2

    z0 = torch.rand(amplitude.shape)
    z0 = z0 / torch.norm(z0)
    z0 = power_method_for_spectral_init(z0, 50)

    # scale eigenvector
    z0 *= torch.sqrt(intensity.mean())
    return z0

# =================
# Helper functions
# =================

def power_method_for_spectral_init(z0: torch.Tensor, intensity: torch.Tensor, iteration: int):
    for _ in range(iteration):
        z0 = utils.ifft2d(intensity * utils.fft2d(z0))
        z0 = z0 / torch.norm(z0)
    return z0
