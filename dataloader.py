import os
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision import transforms

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

def get_valid_png_loader(root, batch_size):
    transform = transforms.Compose([
                transforms.ToTensor()])
    dataset = PNGDataset(root, False, transform)
    data_loader = DataLoader(dataset, batch_size)
    return data_loader
