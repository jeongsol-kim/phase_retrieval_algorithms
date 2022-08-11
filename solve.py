import os
import argparse
import torch
from torchvision.utils import save_image

from dataloader import get_valid_loader
from algorithms import get_algorithm
from utils import crop_center_half, ifft2d, normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='ER')
    parser.add_argument('--num_iterations', type=int, default=2000)
    parser.add_argument('--data_root', type=str, default='/data/FFHQ/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    loader = get_valid_loader('noise_amplitude_dataset', 
                              root=args.data_root,
                              batch_size=args.batch_size,
                              sigma=0.05)

    algorithm = get_algorithm(args.algorithm)

    save_dir = f'results/{args.algorithm}/'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (image, amplitude, support) in enumerate(loader):
            print(f">> Phase retrieval for {i}-th image.")
            image = image.to(device)
            amplitude = amplitude.to(device)
            support = support.to(device)
            recon = algorithm(amplitude, support, args.num_iterations)
            recon = crop_center_half(recon)

            raw_input = torch.real(crop_center_half(ifft2d(amplitude)))
            result = torch.cat([crop_center_half(image), normalize(raw_input), normalize(recon)])
            save_image(normalize(recon), os.path.join(save_dir, f"recon_{i}.png"))

if __name__ == "__main__":
    main()
