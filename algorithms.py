from typing import Tuple
import numpy as np
import torch

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
def error_reduction_algorithm():
    init_phase = generate_random_phase()

@register_algorithm(name='HIO')
def hybrid_input_and_output_algorithm():
    pass


# =================
# Helper functions
# =================

#TODO: function that generate support
#TODO: determine 1) algorithm get support and amplitude 2) algorithm only get amplitude
#TODO: ER algorithm. support constraint and non-negative constraint.

def generate_random_phase(shape: Tuple[int, int]) -> torch.Tensor:
    '''Generate randomized phase [-pi, pi] of given shape.'''
    # TODO: check if uniform distribution is correct.
    random_uniform = torch.rand(shape)  # [0, 1)
    random_phase = (random_uniform - 0.5) * 2. * np.pi
    return random_phase

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
