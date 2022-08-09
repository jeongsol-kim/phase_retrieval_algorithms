import os
import argparse
import torch
from torchvision.utils import save_image

from dataloader import get_valid_loader
from algorithms import get_algorithm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='ER')
    parser.add_argument('--num_iterations', type=int, default=2000)
    parser.add_argument('--data_root', type=str, default='/data/FFHQ/')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    loader = get_valid_loader('amplitude_dataset', root=args.data_root, batch_size=args.batch_size)
    algorithm = get_algorithm(args.algorithm)

    amplitude = next(iter(loader))
    recon = algorithm(amplitude, args.num_iterations)

    print(amplitude.shape, recon.shape)

    os.makedirs('results/', exist_ok=True)
    result = torch.cat([amplitude, recon.abs()])
    save_image(result, 'results/test.png')

if __name__ == "__main__":
    main()
