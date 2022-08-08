from typing import Tuple
import torch
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


def zero_padding_twice(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim not in [3, 4]:
        raise ValueError(f"Expected dimension of input (B, C, H, W) or (C, H, W),\
                           but get {tensor.ndim}-D input.")
    
    height, width = tensor.shape[-2], tensor.shape[-1]
    padding = calculate_half_padding(height, width)
    padding_layer = torch.nn.ZeroPad2d(padding)

    return padding_layer(tensor)

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







