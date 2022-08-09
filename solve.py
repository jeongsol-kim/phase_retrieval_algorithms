import os
import argparse
import torch
from torchvision.utils import save_image

from dataloader import get_valid_loader
from algorithms import get_algorithm
from utils import ifft2d, normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='ER')
    parser.add_argument('--num_iterations', type=int, default=2000)
    parser.add_argument('--data_root', type=str, default='/data/FFHQ/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    loader = get_valid_loader('amplitude_dataset', root=args.data_root, batch_size=args.batch_size)
    algorithm = get_algorithm(args.algorithm)

    with torch.no_grad():
        image, amplitude = next(iter(loader))
        image = image.to(device)
        amplitude = amplitude.to(device)
        recon = algorithm(amplitude, args.num_iterations)
        raw_input = ifft2d(amplitude)
        
    os.makedirs('results/', exist_ok=True)
    result = torch.cat([image, raw_input.abs(), normalize(recon.abs())])
    save_image(result, 'results/test.png')

if __name__ == "__main__":
    main()
