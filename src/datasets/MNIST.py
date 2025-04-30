from .dataset_base import DatasetBase
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


class MNIST(DatasetBase):
    def __init__(self):
        super().__init__(MNIST)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.inverse_transform = transforms.Compose([
            transforms.Normalize((-0.42421358519941355,), (3.2448377581,))
        ])

        self.labels = ("0","1","2","3","4","5","6","7","8","9")