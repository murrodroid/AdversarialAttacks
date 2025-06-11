from .dataset_base import DatasetBase
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


class Cifar10(DatasetBase):

    def __init__(self, root="./data", download=True):
        super().__init__(CIFAR10, path=root, download=download)
        self.labels = (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        self.num_classes = len(self.labels)

        self.train_data = CIFAR10(root=root, train=True, download=download)
        self.test_data = CIFAR10(root=root, train=False, download=download)

    @staticmethod
    def transforms(image):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        return transform(image)

    @staticmethod
    def inverse_transforms(tensor):
        inv_mean = [-m/s for m, s in zip((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))]
        inv_std = [1/s for s in (0.2023,0.1994,0.2010)]

        transform = transforms.Compose(
            [transforms.Normalize(tuple(inv_mean), tuple(inv_std))]
        )
        return transform(tensor)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.train_data) + len(self.test_data)

    def __getitem__(self, idx):
        """Fetch the sample at the given index and apply transforms."""
        # Use training data by default, could be enhanced with train/test split logic
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

    def get_all_labels(self, train=False):
        """Returns a list of all labels for a given split."""
        data = self.train_data if train else self.test_data
        return data.targets
