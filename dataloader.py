import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision import transforms

from utils import fft2d, zero_padding_twice

__DATASETS__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASETS__.get(name, None):
            raise NameError(f"Name {name} is already registered.")
        __DATASETS__[name] = cls
        return cls
    return wrapper

def get_dataset(name: str):
    if __DATASETS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASETS__[name]


def get_valid_loader(dataset_name: str, root: str, batch_size: int):
    transform = transforms.Compose([
                transforms.ToTensor()])
    dataset = get_dataset(dataset_name)(root, False, transform)
    data_loader = DataLoader(dataset, batch_size)
    return data_loader


@register_dataset(name='png_dataset')
class PNGDataset(VisionDataset):
    def __init__(self, root: str, train:bool, transform):
        super().__init__(root=root)
        self.transform = transform
        stage = "train" if train else "valid"
        self.image_paths = glob(os.path.join(root, stage, '*.png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        image = Image.open(fp=img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image

@register_dataset(name='amplitude_dataset')
class AmplitudeDataset(PNGDataset):
    def __getitem__(self, index):
        image = super().__getitem__(index)
        
        support = torch.ones_like(image)
        image = zero_padding_twice(image)
        support = zero_padding_twice(support)
        
        fft_image = fft2d(image)
        amplitude = fft_image.abs()

        return image, amplitude, support

