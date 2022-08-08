from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

class CustomDataset(VisionDataset):
    def __init__(self, root: str):
        super().__init__(root=root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index]


def get_loader(root, train):
    pass
