from typing import Tuple, Callable
import torch
from torch.nn.functional import mse_loss
import numpy as np
from tqdm import tqdm

import utils
from initializer import get_initializer

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
    
    # initial guess
    init_fn = get_initializer('gaussian')
    random_phase = init_fn(amplitude.shape).to(amplitude.device)
    G = amplitude * torch.exp(1j * random_phase * 2 * np.pi)

    pbar = tqdm(range(iteration), miniters=100)
    for i in pbar:
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = torch.real(utils.ifft2d(G_prime))
        g = apply_image_constraint(g_prime, support)
        G = utils.fft2d(g)

        loss = mse_loss(G.abs(), amplitude)
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        pbar.set_postfix({'MSE': loss.item()}, refresh=False)
    
    g = torch.real(utils.ifft2d(G))
    final_loss = mse_loss(G.abs(), amplitude)
    return g, final_loss

@register_algorithm(name="HIO")
def hybrid_input_output_algorithm(amplitude: torch.Tensor, support: torch.Tensor, iteration: int):

    # initial guess
    init_fn = get_initializer('gaussian')
    random_phase = init_fn(amplitude.shape).to(amplitude.device)
    G = amplitude * torch.exp(1j * random_phase * 2 * np.pi)
    g = torch.real(utils.ifft2d(G))

    pbar = tqdm(range(iteration), miniters=100)
    for i in pbar:
        G_prime = apply_fourier_constraint(G, amplitude)
        g_prime = torch.real(utils.ifft2d(G_prime))
        
        g = apply_image_constraint_hio(g_prime, g, support)
        G = utils.fft2d(g)

        loss = mse_loss(G.abs(), amplitude)
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        pbar.set_postfix({'MSE': loss.item()}, refresh=False)

    g = torch.real(utils.ifft2d(G))
    final_loss = mse_loss(G.abs(), amplitude)
    return g, final_loss


@register_algorithm(name="OSS")
def oversampling_smoothness_algorithm(amplitude: torch.Tensor, support: torch.Tensor, iteration: int):
    '''https://arxiv.org/ftp/arxiv/papers/1211/1211.4519.pdf'''

    # prepare alpha for gaussian filter (following the paper)
    n_filters = 10
    x_alpha = torch.linspace(amplitude.size(-2), 0.05 * amplitude.size(-2), n_filters)
    y_alpha = torch.linspace(amplitude.size(-1), 0.05 * amplitude.size(-1), n_filters)
    
    # initial guess
    init_fn = get_initializer('gaussian')
    random_phase = init_fn(amplitude.shape).to(amplitude.device)
    G = amplitude * torch.exp(1j * random_phase * 2 * np.pi)
    g = torch.real(utils.ifft2d(G))

    # initial loss for choosing best recon.
    loss = 1e9

    for i in range(n_filters):
        best = {'recon': G, 'loss': loss}
        pbar = tqdm(range(int(iteration/n_filters)), miniters=100)
        for _ in pbar:
            G_prime = apply_fourier_constraint(G, amplitude)
            g_prime = torch.real(utils.ifft2d(G_prime))
            gaussian_filter = generate_gaussian_filter(amplitude, x_alpha[i], y_alpha[i])
            g = apply_image_constraint_oss(g_prime, g, support, gaussian_filter)
            G = utils.fft2d(g)

            loss = mse_loss(G.abs(), amplitude)
            if torch.isnan(loss):
                loss = torch.tensor(np.inf)
            if best.get('loss') > loss:
                best.update({'recon' : G, 'loss': loss})

            pbar.set_description(f"Iteration {i+1}", refresh=False)
            pbar.set_postfix({"MSE" : loss.item()})

        G = best.get('recon')

    g = torch.real(utils.ifft2d(G))
    final_loss = mse_loss(G.abs(), amplitude)
    return g, final_loss


@register_algorithm(name="WF")
def wirtinger_flow_algorithm(amplitude: torch.Tensor, support: torch.Tensor, iteration: int):

    # initialize step size schedule
    tau0 = 330
    mu_max = 0.4
    mu = [min(1-np.exp(-t/tau0), mu_max) for t in range(1, iteration+1)]

    # initial guess
    init_fn = get_initializer('spectral')
    z = init_fn(amplitude, power_iteration=500)

    pbar = tqdm(range(iteration), miniters=100)
    for i in pbar:
        estimate = utils.fft2d(z)
        temp = (estimate.abs() ** 2 - amplitude ** 2) * estimate
        grad = utils.ifft2d(temp) / torch.numel(temp)

        z = z - mu[i] / (amplitude ** 2).mean() * grad
        
        loss = mse_loss(utils.fft2d(z).abs(), amplitude)
        pbar.set_description(f"Iteration {i+1}", refresh=False)
        pbar.set_postfix({'MSE': loss.item()}, refresh=False)

    final_loss = mse_loss(utils.fft2d(z).abs(), amplitude)
    return z.abs(), final_loss

# =================
# Helper functions
# =================

def generate_gaussian_filter(amplitude: torch.Tensor, x_alpha: float, y_alpha: float):
    x = torch.arange(-round((amplitude.size(-2) - 1) / 2), round((amplitude.size(-2) -1) / 2), amplitude.size(-2))
    y = torch.arange(-round((amplitude.size(-1) - 1) / 2), round((amplitude.size(-1) -1) / 2), amplitude.size(-1))
    X, Y = torch.meshgrid(x, y)

    gaussian_filter = torch.exp(-0.5 * ((X/x_alpha) ** 2)) * torch.exp(-0.5 * ((Y/y_alpha) ** 2))

    # normalize and repeat filter
    gaussian_filter /= gaussian_filter.max()
    gaussian_filter = gaussian_filter.expand(amplitude.shape)
    gaussian_filter = gaussian_filter.to(amplitude.device)
        
    return gaussian_filter

def generate_random_phase(amplitude: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    random_uniform = torch.rand(amplitude.shape).to(support.device)
    random_phase = random_uniform * support
    return random_phase

def apply_image_constraint(obj, support):
    support = support * generate_non_negative_support(obj)
    obj = obj * support
    return obj

def apply_image_constraint_hio(obj, prev_obj, support, beta=0.9):
    support = support * generate_non_negative_support(obj)
    in_support = obj * support
    out_support = (prev_obj - beta * obj) * (1-support)
    return in_support + out_support


def apply_image_constraint_oss(obj, prev_obj, support, gaussian_filter):
    new_obj = apply_image_constraint_hio(obj, prev_obj, support)
    
    # for oss conditioning, don't use non-negative constraint.
    in_support = new_obj * support
    out_support = torch.real(utils.ifft2d(utils.fft2d(new_obj) * gaussian_filter)) * (1-support)
    return in_support + out_support


def generate_non_negative_support(obj: torch.Tensor) -> torch.Tensor:
    nn_support = torch.ones_like(obj)
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
