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

    @abstractmethod
    def get_all_labels(self, train=False):
        """Returns a list of all labels for a given split."""
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


class AdvDataset:

    def __init__(
        self, dataset, num_classes, num_images_per_class, pairing_mode="all_targets"
    ):
        self.dataset = dataset
        self.pairing_mode = pairing_mode

        labels = dataset.get_all_labels(train=False)
        by_class = {c: [] for c in range(num_classes)}

        for i, lbl in enumerate(labels):
            if len(by_class[lbl]) < num_images_per_class:
                by_class[lbl].append(i)

        if pairing_mode == "all_targets":
            self.samples = [
                (src, tgt, idx)
                for src in range(num_classes)
                for tgt in range(num_classes)
                if tgt != src
                for idx in by_class[src]
            ]
        elif pairing_mode == "random_target":
            self.samples = []
            for src in range(num_classes):
                for idx in by_class[src]:
                    possible_targets = [tgt for tgt in range(num_classes) if tgt != src]
                    tgt = random.choice(possible_targets)
                    self.samples.append((src, tgt, idx))
        else:
            raise ValueError(
                f"Invalid pairing_mode: {pairing_mode}. Must be 'all_targets' or 'random_target'"
            )

        print(f"Using pairing mode: {pairing_mode}")
        print("Preloading tensors...")
        self.cached_tensors = {}
        unique_indices = {idx for _, _, idx in self.samples}

        if hasattr(dataset, "test_data") and hasattr(dataset.test_data, "data"):
            self._cache_from_raw_data(dataset, unique_indices)
        else:
            self._cache_from_dataset(dataset, unique_indices)

        print(
            f"Cached {len(self.cached_tensors)} tensors, created {len(self.samples)} samples"
        )

    def _cache_from_raw_data(self, dataset, indices):
        import torchvision.transforms.functional as TF

        data = dataset.test_data.data  # (N, H, W, C) numpy array
        for idx in indices:
            img_np = data[idx]  # (H, W, C)
            tensor = (
                torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            )  # (C, H, W)
            tensor = TF.normalize(
                tensor, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
            self.cached_tensors[idx] = tensor

    def _cache_from_dataset(self, dataset, indices):
        """Cache tensors using dataset's get_by_index method"""
        for idx in indices:
            tensor = dataset.get_by_index(idx, train=False)["tensor"].squeeze(0)
            self.cached_tensors[idx] = tensor
