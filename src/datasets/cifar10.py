from .dataset_base import DatasetBase
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


class Cifar10(DatasetBase):
    def __init__(self):
        super().__init__(CIFAR10)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
        ])
        
        inv_mean = [-m/s for m, s in zip((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))]
        inv_std = [1/s for s in (0.2023,0.1994,0.2010)]

        self.inverse_transform = transforms.Compose([
            transforms.Normalize(tuple(inv_mean), tuple(inv_std))
        ])