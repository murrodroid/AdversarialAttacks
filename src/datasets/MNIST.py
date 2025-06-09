from .dataset_base import DatasetBase
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as TorchMNIST


class MNIST(DatasetBase):

    def __init__(self, root="./data", download=True):
        super().__init__(TorchMNIST, path=root, download=download)
        self.labels = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        self.num_classes = len(self.labels)

        self.train_data = TorchMNIST(root=root, train=True, download=download)
        self.test_data = TorchMNIST(root=root, train=False, download=download)

    @staticmethod
    def transforms(image):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return transform(image)

    @staticmethod
    def inverse_transforms(tensor):
        transform = transforms.Compose(
            [transforms.Normalize((-0.42421358519941355,), (3.2448377581,))]
        )
        return transform(tensor)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.train_data) + len(self.test_data)

    def __getitem__(self, idx):
        """Fetch the sample at the given index and apply transforms."""
        if idx < len(self.train_data):
            img, label = self.train_data[idx]
        else:
            img, label = self.test_data[idx - len(self.train_data)]

        tensor = self.__class__.transforms(img)
        return tensor, label

    def get_by_index(self, index, train=False):
        """Returns a sample dict given its index."""
        data = self.train_data if train else self.test_data
        img, label = data[index]
        tensor = self.__class__.transforms(img).unsqueeze(0)  # [1, C, H, W]
        return {"tensor": tensor, "label": label, "index": index}

    def get_indices_from_class(self, class_idx, train=False, num_images=None):
        """Get indices of samples belonging to a specific class."""
        data = self.train_data if train else self.test_data
        indices = [i for i, (_, label) in enumerate(data) if label == class_idx]
        if num_images is not None:
            indices = indices[:num_images]
        return indices
