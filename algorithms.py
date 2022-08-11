from typing import Tuple, Callable
import torch
from torch.nn.functional import mse_loss
import numpy as np
from tqdm import tqdm

import utils

__ALGORITHMS__ = {}

def register_algorithm(name: str):
    def wrapper(func):
        if __ALGORITHMS__.get(name, None):
            raise NameError(f"Name {name} is already registered.")
        __ALGORITHMS__[name] = func
        return func
    return wrapper

def get_algorithm(name: str) -> Callable:
    if __ALGORITHMS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __ALGORITHMS__[name]

@register_algorithm(name='ER')
def error_reduction_algorithm(amplitude: torch.Tensor, support: torch.Tensor, iteration: int):
    
    # initial guess
    random_phase = torch.rand(amplitude.shape).to(amplitude.device)
    G = amplitude * torch.exp(1j * random_phase * 2 * np.pi)

    pbar = tqdm(range(iteration), miniters=100)
    for i in pbar:
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = torch.real(utils.ifft2d(G_prime))
        g = apply_image_constraint(g_prime, support)
        G = utils.fft2d(g)

        loss = mse_loss(G.abs(), amplitude)
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        pbar.set_postfix({'MSE': loss.item()}, refresh=False)
    
    g = torch.real(utils.ifft2d(G))
    return g

@register_algorithm(name="HIO")
def hybrid_input_output_algorithm(amplitude: torch.Tensor, support: torch.Tensor, iteration):

    # initial guess
    random_phase = torch.rand(amplitude.shape).to(amplitude.device)
    G = amplitude * torch.exp(1j * random_phase * 2 * np.pi)
    g = torch.real(utils.ifft2d(G))

    pbar = tqdm(range(iteration), miniters=100)
    for i in pbar:
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = torch.real(utils.ifft2d(G_prime))
        
        g = apply_image_constraint_hio(g_prime, g, support)
        G = utils.fft2d(g)

        loss = mse_loss(G.abs(), amplitude)
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        pbar.set_postfix({'MSE': loss.item()}, refresh=False)

    g = torch.real(utils.ifft2d(G))
    return g

# =================
# Helper functions
# =================

#TODO: ER algorithm. support constraint and non-negative constraint.

def generate_random_phase(padded_amplitude: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    random_uniform = torch.rand(padded_amplitude.shape).to(support.device)
    random_phase = random_uniform * support
    return random_phase

def apply_image_constraint(obj, support):
    support = support * generate_non_negative_support(obj)
    obj = obj * support
    return obj

def apply_image_constraint_hio(obj, prev_obj, support, beta=0.9):
    support = support * generate_non_negative_support(obj)
    in_support = obj * support
    out_support = (prev_obj - beta * obj) * (1-support)
    return in_support + out_support

def generate_non_negative_support(obj: torch.Tensor) -> torch.Tensor:
    nn_support = torch.ones_like(obj)
    nn_support[obj < 0 ] = 0
    return nn_support.to(obj.device)

def apply_fourier_constraint(fft_obj, measured_amplitude):
    substituted_obj = substitute_amplitude(fft_obj, measured_amplitude)
    return substituted_obj


def substitute_amplitude(complex_obj: torch.Tensor, measured_amplitude: torch.Tensor) -> torch.Tensor:
    """Substitute amplitude of complex object with measured ampiltude.

    Args:
        complex_obj (torch.Tensor): Complex object that has amplitude and phase.
        measured_amplitude (torch.Tensor): Measured amplitude.

    Returns:
        torch.Tensor: Substituted complex object that has the same phase with input data.
    """
    estimated_amplitude = complex_obj.abs()
    substituted_obj = complex_obj / estimated_amplitude * measured_amplitude
    return substituted_obj
