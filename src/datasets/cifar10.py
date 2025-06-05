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

        self.labels = ("airplane", "automobile","bird","cat","deer","dog","frog","horse","ship","truck")

        self.train_data = CIFAR10(root="./data", train=True, download=True)
        self.test_data = CIFAR10(root="./data", train=False, download=True)

        
    def get_by_index(self, index, train=False):
        """Returns a sample dict given its index."""
        data = self.train_data if train else self.test_data
        img, label = data[index]
        tensor = self.transform(img).unsqueeze(0)  # [1, C, H, W]
        return {"tensor": tensor, "label": label, "index": index}
    
    def get_indices_from_class(self, class_id, train=False, num_images=2):
        data = self.train_data if train else self.test_data
        indices = [i for i, (_, label) in enumerate(data) if label == class_id]
        return indices[:num_images]