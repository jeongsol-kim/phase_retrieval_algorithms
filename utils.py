from typing import Tuple
import torch
from torch.nn import functional as F
from torch.fft import fft2, ifft2

def fft2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim not in [3, 4]:
        raise ValueError(f"Expected dimension of input (B, C, H, W) or (C, H, W),\
                           but get {tensor.ndim}-D input.")
    return fft2(tensor)


def ifft2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim not in [3, 4]:
        raise ValueError(f"Expected dimension of input (B, C, H, W) or (C, H, W),\
                           but get {tensor.ndim}-D input.")
    return ifft2(tensor)


def split_amplitude_phase(complex_obj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amplitude = complex_obj.abs()
    phase = complex_obj.angle()
    return amplitude, phase

def combine_amplitude_phase(amplitude: torch.Tensor, phase: torch.Tensor):
    complex_obj = torch.polar(amplitude, phase)
    return complex_obj

def zero_padding_twice(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim not in [3, 4]:
        raise ValueError(f"Expected dimension of input (B, C, H, W) or (C, H, W),\
                           but get {tensor.ndim}-D input.")
    
    height, width = tensor.shape[-2], tensor.shape[-1]
    padding = calculate_half_padding(height, width)
    # padding_layer = torch.nn.ZeroPad2d(padding)
    # return padding_layer(tensor)
    return F.pad(tensor, padding, 'constant', 0)

def calculate_half_padding(height: int, width: int) -> Tuple[int, int, int, int]:
    """Calculate half size padding given height and width.
    If height is an odd value, add 1 to bottom padding.
    If width is an odd value, add 1 to right padding.

    Args:
        height (int): height of an image
        width (int): width of an image

    Returns:
        Tuple[int, int, int, int]: half size padding, (left, right, top, bottom).
    """
    
    pad_top = height // 2
    pad_bottom = height - pad_top
    pad_left = width // 2
    pad_right = width - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)

def crop_center_half(tensor: torch.Tensor) -> torch.Tensor:
    height, width = tensor.shape[-2], tensor.shape[-1]
    padding = calculate_half_padding(height//2, width//2)
    return tensor[..., padding[2]:-padding[3], padding[0]:-padding[1]]

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    max_pixels = tensor.max(dim=-2, keepdim=True)[0]
    max_pixels = max_pixels.max(dim=-1, keepdim=True)[0]
    min_pixels = tensor.min(dim=-2, keepdim=True)[0]
    min_pixels = min_pixels.min(dim=-1, keepdim=True)[0]
    return (tensor-min_pixels) / (max_pixels-min_pixels)

