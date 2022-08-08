import numpy as np
import torch
from algorithms import generate_random_phase, substitute_amplitude


def test_generate_random_phase_scale():
    shape = (128, 128)
    for _ in range(10):
        random_phase = generate_random_phase(shape)
        min_phase = random_phase.min()
        max_phase = random_phase.max()
        assert min_phase.numpy() >= -np.pi and max_phase.numpy() <= np.pi

def test_generate_random_phase_shape():
    shape = (128, 128)
    random_phase = generate_random_phase(shape)
    assert random_phase.shape == shape


def test_substitute_amplitude():
    init_amp = torch.ones((128, 128))
    init_phase = torch.ones((128, 128)) * np.pi
    true_amp = init_amp * 0.5

    complex_obj = init_amp * torch.exp(1j*init_phase)
    substituted_obj = substitute_amplitude(complex_obj, true_amp)

    assert (substituted_obj.abs() == true_amp).all()


