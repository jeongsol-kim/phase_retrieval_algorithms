import torch
import pytest

from utils import calculate_half_padding, fft2d, ifft2d, zero_padding_twice

def test_fft2d_shape():
    dummy = torch.ones((1, 3, 256, 256))
    fft_dummy = fft2d(dummy)
    assert fft_dummy.shape == dummy.shape

def test_ifft2d_identical():
    dummy = torch.ones((1, 3, 256, 256)).to(torch.complex64)
    fft_dummy = fft2d(dummy)
    recon_dummy = ifft2d(fft_dummy)
    assert torch.allclose(dummy, recon_dummy)

def test_fft2d_bad_shape_error():
    dummy = torch.ones((256, 256))
    
    with pytest.raises(ValueError):
        _ = fft2d(dummy)

def test_ifft2d_bad_shape_error():
    dummy = torch.ones((1, 3, 256, 256))
    fft_dummy = fft2d(dummy)
    
    with pytest.raises(ValueError):
        _ = ifft2d(fft_dummy[0,:,:,0])

def test_zero_padding_twice_shape():
    dummy = torch.ones((1, 3, 128, 128))
    padded_dummy = zero_padding_twice(dummy)
    assert padded_dummy.shape == (1, 3, 256, 256)

def test_zero_padding_twice_values():
    dummy = torch.ones((1, 3, 128, 128))
    padded_dummy = zero_padding_twice(dummy)
    assert padded_dummy.sum() == 3*128*128
    
def test_zero_padding_twice_bad_shape_error():
    dummy = torch.ones((256, 256))
    with pytest.raises(ValueError):
        _ = zero_padding_twice(dummy)

def test_calculate_half_padding_even_case():
    height = 32
    width = 32
    padding = calculate_half_padding(height, width)
    assert padding == (16, 16, 16, 16)

def test_calculate_half_padding_odd_case():
    height = 31
    width = 32
    padding = calculate_half_padding(height, width)
    assert padding == (16, 16, 15, 16)



