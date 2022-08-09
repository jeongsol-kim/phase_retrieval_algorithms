from typing import Tuple
import numpy as np
import torch

import utils

__ALGORITHMS__ = {}

def register_algorithm(name: str):
    def wrapper(func):
        if __ALGORITHMS__.get(name, None):
            raise NameError(f"Name {name} is already registered.")
        __ALGORITHMS__[name] = func
        return func
    return wrapper

def get_algorithm(name: str):
    if __ALGORITHMS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __ALGORITHMS__[name]

@register_algorithm(name='ER')
def error_reduction_algorithm(amplitude: torch.Tensor, iteration: int, start_domain: str = 'fourier'):
    padded_amplitude, support = generate_support(amplitude)
    init_phase = generate_random_phase(padded_amplitude, support)

    for i in range(iteration):
        apply_image_constraint()
        apply_fourier_constraint()
    
    

@register_algorithm(name='HIO')
def hybrid_input_and_output_algorithm(amplitude: torch.Tensor, iteration: int, start_domain: str = 'fourier'):
    pass


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


def apply_image_constraint(amplitude, phase, support):
    pass


def apply_fourier_constraint(amplitude, phase, measured_amplitude):
    pass


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
