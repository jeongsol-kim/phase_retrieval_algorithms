import pytest
import torch
from dataloader import get_valid_loader

def test_get_loader_undefined_name_error():
    root = '/data/FFHQ/'
    batch_size = 1
    
    with pytest.raises(NameError):
        _ = get_valid_loader('wired_name', root, batch_size)

def test_png_loader_data_shape():
    root = '/data/FFHQ/'
    batch_size = 2

    loader = get_valid_loader('png_dataset', root, batch_size)
    image = next(iter(loader))
    assert image.shape == (batch_size, 3, 256, 256)

def test_png_loader_data_scale():
    root = '/data/FFHQ/'
    batch_size = 1

    loader = get_valid_loader('png_dataset', root, batch_size)
    image = next(iter(loader))
    assert image.min() >= 0.0 and image.max() <= 1.0

def test_amplitude_loader_data_shape():
    root = '/data/FFHQ/'
    batch_size = 2

    loader = get_valid_loader('amplitude_dataset', root, batch_size)
    image = next(iter(loader))
    assert image.shape == (batch_size, 3, 256, 256)


def test_amplitude_loader_data_type():
    root = '/data/FFHQ/'
    batch_size = 1

    loader = get_valid_loader('amplitude_dataset', root, batch_size)
    image = next(iter(loader))
    assert image.dtype == torch.float32
