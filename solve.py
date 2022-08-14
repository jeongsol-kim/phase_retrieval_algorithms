from collections.abc import Callable
import os
import logging
import time
import argparse
import torch
from torchvision.utils import save_image

from dataloader import get_valid_loader
from algorithms import get_algorithm
from utils import crop_center_half, ifft2d, normalize


def set_logger(log_path: str):
    logger = logging.getLogger()
    
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s >> %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def make_directories(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'measure_images/'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'measure_kspace/'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'recon/'), exist_ok=True)


def run(algorithm, dataloader, args):
    # set device, loader, and algorithm
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # prepare directories and logger
    save_dir = f'results/{args.algorithm}/'
    make_directories(save_dir)
    log_path = os.path.join(save_dir, 'stdout.log')
    logger = set_logger(log_path)

    # RUN
    for i, (image, amplitude, support) in enumerate(dataloader):
        logger.info("Phase retrieval for %d-th image.", i)

        image = image.to(device)
        amplitude = amplitude.to(device)
        support = support.to(device)
    
        best_recon = {'recon': None, 'loss':1e9}
        average_time = 0
        
        for n in range(args.num_repeats):
            logger.info("Repeat Time %d.", n+1)
            start_time = time.time()
            recon, loss = algorithm(amplitude, support, args.num_iterations)
            run_time = time.time() - start_time
            
            average_time += run_time
            
            if torch.isnan(loss):
                loss = torch.tensor([100.0]).to(loss.device)
            if best_recon['loss'] > loss:
                best_recon.update({'recon': recon, 'loss':loss})

        average_time = average_time / args.num_repeats
        logger.info(f"Best loss: {round(best_recon['loss'].item(), 3)}")
        logger.info(f"Average time: {average_time}")
        
        recon = crop_center_half(best_recon['recon'])

        measurement_image = torch.real(crop_center_half(ifft2d(amplitude)))
        save_image(normalize(recon), os.path.join(save_dir, f"recon/recon_{i}.png"))
        save_image(normalize(measurement_image), os.path.join(save_dir, f"measure_images/measure_img_{i}.png"))
        save_image(normalize(amplitude), os.path.join(save_dir, f"measure_kspace/measurement_{i}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='ER')
    parser.add_argument('--num_iterations', type=int, default=2000)
    parser.add_argument('--num_repeats', type=int, default=4)
    parser.add_argument('--data_root', type=str, default='/data/FFHQ/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    loader = get_valid_loader('noise_amplitude_dataset',
                              root=args.data_root,
                              batch_size=args.batch_size,
                              sigma=0.05)

    algorithm = get_algorithm(args.algorithm)

    with torch.no_grad():
        run(algorithm, loader, args)   

if __name__ == "__main__":
    main()
