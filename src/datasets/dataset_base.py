from abc import ABC, abstractmethod
import torch
import random
from torchvision import transforms


class DatasetBase(ABC):

    def __init__(self, dataset=None, path="./data", download=True):
        self.download = download
        self.path = path
        self._dataset = dataset
        self.labels = None  # Should be set by subclasses
        self.num_classes = None  # Should be set by subclasses

    @staticmethod
    @abstractmethod
    def transforms(image):
        """Apply dataset-specific transforms to an image."""
        pass

    @staticmethod
    @abstractmethod
    def inverse_transforms(tensor):
        """Apply inverse transforms to convert normalized tensor back to [0,1] range."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Fetch the sample at the given index and apply transforms."""
        pass

    @abstractmethod
    def get_by_index(self, index, train=False):
        """Returns a sample dict given its index.

        Args:
            index (int): The index of the sample
            train (bool): Whether to use training or validation split

        Returns:
            dict: {"tensor": torch.Tensor, "label": int, "index": int}
        """
        pass

    @abstractmethod
    def get_indices_from_class(self, class_idx, train=False, num_images=None):
        """Get indices of samples belonging to a specific class.

        Args:
            class_idx (int): The class index
            train (bool): Whether to use training or validation split
            num_images (int, optional): Maximum number of images to return

        Returns:
            list: List of indices
        """
        pass

    def get_class_name(self, class_id):
        """Get the class name for a given class ID."""
        if self.labels is None:
            raise NotImplementedError("Labels not defined for this dataset")
        return self.labels[class_id]

    def get_sample_from_class(self, class_idx, train=True, num_images=1):
        """Get sample(s) from a specific class.

        Args:
            class_idx (int): The class index
            train (bool): Whether to use training or validation split
            num_images (int): Number of images to return

        Returns:
            list: List of sample dictionaries
        """
        try:
            indices = self.get_indices_from_class(
                class_idx, train=train, num_images=num_images
            )

            results = []
            for idx in indices:
                try:
                    sample = self.get_by_index(idx, train=train)
                    results.append(sample)
                except Exception as e:
                    print(f"[Debug Dataset] Error processing index {idx}: {e}")

            return results

        except Exception as e:
            print(
                f"[Debug Dataset] Error in get_sample_from_class for class {class_idx}: {e}"
            )
            import traceback

            traceback.print_exc()
            return []
