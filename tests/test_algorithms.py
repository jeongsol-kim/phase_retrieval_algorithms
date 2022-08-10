import numpy as np
import torch
from algorithms import generate_random_phase, generate_support, substitute_amplitude


def test_generate_random_phase_scale():
    shape = (3, 128, 128)
    amplitude = torch.ones(shape)
    support = torch.zeros(shape)
    support[:, 32:96, 32:96] = 1

    for _ in range(10):
        random_phase = generate_random_phase(amplitude, support)
        min_phase = random_phase.min()
        max_phase = random_phase.max()
        assert min_phase.numpy() >= -np.pi and max_phase.numpy() <= np.pi

def test_generate_random_phase_shape():
    shape = (3, 128, 128)
    amplitude = torch.ones(shape)
    support = torch.zeros(shape)
    support[:, 32:96, 32:96] = 1
    random_phase = generate_random_phase(amplitude, support)
    assert random_phase.shape == shape

def test_generate_support_shape():
    dummy = torch.zeros((3, 128, 128))
    support = generate_support(dummy)
    assert dummy.shape == support.shape

def test_generate_support_value():
    dummy = torch.ones((3, 128, 128))
    support = generate_support(dummy)
    assert support.sum() == 3 * 128 * 128

def test_substitute_amplitude():
    init_amp = torch.ones((128, 128))
    init_phase = torch.ones((128, 128)) * np.pi
    true_amp = init_amp * 0.5

    complex_obj = init_amp * torch.exp(1j*init_phase)
    substituted_obj = substitute_amplitude(complex_obj, true_amp)

    assert (substituted_obj.abs() == true_amp).all()


