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
def error_reduction_algorithm(amplitude: torch.Tensor, iteration: int):
    padded_amplitude, support = generate_support(amplitude)
    init_phase = generate_random_phase(padded_amplitude, support)

    fft_obj = utils.combine_amplitude_phase(padded_amplitude, init_phase)
    obj = utils.ifft2d(fft_obj)

    pbar = tqdm(range(iteration), miniters=100)

    for i in pbar:
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        obj = apply_image_constraint(obj, support)
        fft_obj = utils.fft2d(obj)
        fft_obj = apply_fourier_constraint(fft_obj, padded_amplitude)
        obj = utils.ifft2d(fft_obj)

    reconstructed = utils.crop_center_half(obj)
    return reconstructed


@register_algorithm(name='HIO')
def hybrid_input_and_output_algorithm(amplitude: torch.Tensor, iteration: int, start_domain: str = 'fourier'):
    padded_amplitude, support = generate_support(amplitude)
    init_phase = generate_random_phase(padded_amplitude, support)

    fft_obj = utils.combine_amplitude_phase(padded_amplitude, init_phase)
    obj = utils.ifft2d(fft_obj)
    obj_prime = obj
    pbar = tqdm(range(iteration), miniters=100)

    for i in pbar:
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        obj = apply_image_constraint_hio(obj_prime, obj, support)
        fft_obj = utils.fft2d(obj)
        fft_obj = apply_fourier_constraint(fft_obj, padded_amplitude)
        obj_prime= utils.ifft2d(fft_obj)

    reconstructed = utils.crop_center_half(obj_prime)
    return reconstructed

# =================
# Helper functions
# =================

#TODO: ER algorithm. support constraint and non-negative constraint.

def generate_random_phase(padded_amplitude: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    random_uniform = torch.rand(padded_amplitude.shape)
    random_phase = random_uniform * support
    return random_phase


def generate_support(amplitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    padded_amplitude = utils.zero_padding_twice(amplitude)
    support_ones = torch.ones_like(amplitude)
    support = utils.zero_padding_twice(support_ones)
    return padded_amplitude, support


def apply_image_constraint(obj, support):
    obj = obj * support
    return obj

def apply_image_constraint_hio(obj, obj_prime, support, beta=0.9):
    in_support = obj_prime * support
    out_support = (obj - beta * obj_prime) * (1-support)
    return in_support + out_support

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
