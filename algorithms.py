from typing import Tuple, Callable
import torch
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
    g = torch.rand(utils.fft2d(amplitude).shape).to(amplitude.device)

    pbar = tqdm(range(iteration), miniters=100)
    for i in pbar:
        G = utils.fft2d(g)
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = utils.ifft2d(G_prime)
        g = apply_image_constraint(g_prime, support)

        loss = ((G.abs() - amplitude) ** 2).sum() ** 0.5
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        pbar.set_postfix({'MSE': loss.item()}, refresh=False)

    return g

@register_algorithm(name="HIO")
def hybrid_input_output_algorithm(amplitude: torch.Tensor, support: torch.Tensor, iteration):
    g = torch.rand(utils.fft2d(amplitude).shape).to(amplitude.device)

    pbar = tqdm(range(iteration), miniters=100)
    for i in pbar:
        G = utils.fft2d(g)
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = utils.ifft2d(G_prime)
        g = apply_image_constraint_hio(g, g_prime, support)

        loss = ((G.abs() - amplitude) ** 2).sum() ** 0.5
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        pbar.set_postfix({'MSE': loss.item()}, refresh=False)

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

def apply_image_constraint_hio(obj, obj_prime, support, beta=0.9):
    support = support * generate_non_negative_support(obj)
    in_support = obj_prime * support
    out_support = (obj - beta * obj_prime) * (1-support)
    return in_support + out_support

def generate_non_negative_support(obj: torch.Tensor) -> torch.Tensor:
    nn_support = torch.ones_like(obj)

    if obj.dtype == torch.complex64:
        nn_support[torch.real(obj) < 0] = 0
    else:
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
