import torch
import os
import shutil
import tempfile
import numpy as np
from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as T
from pathlib import Path

from .dataset_base import DatasetBase


def get_repo_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))


# Downloading imagenet100 dataset from huggingface
# saving it locally then to a folder in the git repository and deleting the temporary cache folder
def load_imagenet100():
    repo_root = get_repo_root()
    final_dataset_path = os.path.join(repo_root, "data", "imagenet100")

    if os.path.exists(final_dataset_path):
        print("ImageNet100 dataset already exists in the expected location. Skipping download.")
        return final_dataset_path

    os.makedirs(os.path.dirname(final_dataset_path), exist_ok=True)
    print(f"Ensured data directory exists at: {os.path.dirname(final_dataset_path)}")

    temp_cache_dir = os.path.join(tempfile.gettempdir(), "cache_imagenet100")
    os.makedirs(temp_cache_dir, exist_ok=True)
    print(f"Ensured cache directory exists at: {temp_cache_dir}")

    # loading the dataset into a temporary cache directory to avoid issues
    print(f"Attempting to load ImageNet100 using temporary cache: {temp_cache_dir}")
    ds_dict = load_dataset("clane9/imagenet-100", cache_dir=temp_cache_dir)
    print("Dataset loaded into temporary cache successfully.")

    # Copying the dataset to the final location
    print(f"Saving the dataset to its final location: {final_dataset_path}")
    ds_dict.save_to_disk(final_dataset_path)
    print(f"ImageNet100 dataset successfully saved to {final_dataset_path}")

    # Remove the temporary cache directory to avoid storing unnecessary data
    if os.path.exists(temp_cache_dir):
        print(f"Removing temporary cache directory: {temp_cache_dir}")
        try:
            shutil.rmtree(temp_cache_dir)
            print("Temporary cache directory removed successfully.")
        except Exception as e:
            print(f"Error removing temporary cache directory {temp_cache_dir}: {e}")
    return final_dataset_path


class ImageNet100(DatasetBase):
    def __init__(self, root_dir=None, train=False, validation=False):
        super().__init__()

        if root_dir is None:
            root_dir = load_imagenet100()

        self.root_dir = root_dir
        split_name = "train" if train else "validation"
        self.dire = os.path.join(self.root_dir, split_name)
        self.dataset = load_from_disk(self.dire)

        # Store train/validation state
        self.is_train = train
        self.is_validation = validation

        self.train_transforms = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transforms = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.num_classes = self.dataset.features["label"].num_classes
        self.labels = list(range(self.num_classes))

    @staticmethod
    def transforms(image):
        """Apply dataset-specific transforms to an image (validation transforms by default)."""
        transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Ensure image is RGB, as required by standard models
        return transform(image.convert("RGB"))

    @staticmethod
    def transforms_train(image):
        """Apply training-specific transforms to an image."""
        transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Ensure image is RGB, as required by standard models
        return transform(image.convert("RGB"))

    @staticmethod
    def inverse_transforms(tensor):
        """Convert normalized ImageNet tensor back to [0,1] range."""
        inv_mean = [
            -m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        inv_std = [1 / s for s in [0.229, 0.224, 0.225]]

        transform = T.Compose([T.Normalize(tuple(inv_mean), tuple(inv_std))])
        return transform(tensor)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index and applies transforms.
        """
        # 5. Get a single item from the dataset. It's a dictionary.
        # The default keys are 'image' and 'label'.
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        # 6. Apply transforms to the PIL image.
        transforms_to_use = (
            self.train_transforms if self.is_train else self.val_transforms
        )
        if transforms_to_use:
            # Ensure image is RGB, as required by standard models
            image = transforms_to_use(image.convert("RGB"))

        return image, label

    def get_by_index(self, idx, train=False):
        """Returns a sample dict given its index."""
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        # Apply transforms
        if train:
            tensor = self.__class__.transforms_train(image).unsqueeze(0)  # [1, C, H, W]
        else:
            tensor = self.__class__.transforms(image).unsqueeze(0)  # [1, C, H, W]
        return {"tensor": tensor, "label": label, "index": idx}

    def get_indices_from_class(self, class_idx, train=False, num_images=None):
        """Get indices of samples belonging to a specific class."""
        indices = [i for i, item in enumerate(self.dataset) if item["label"] == class_idx]
        if num_images is not None:
            indices = indices[:num_images]
        return indices

    def get_all_labels(self, train=False):
        """Returns a list of all labels for a given split."""
        return self.dataset["label"]


def path_to_imagenet100():
    """
    Finds the ImageNet100 dataset in the expected location.
    Returns the path to the dataset directory.
    """
    repo_root = get_repo_root()
    data_directory = os.path.join(repo_root, "data", "imagenet100")

    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"ImageNet100 dataset not found at {data_directory}")

    return data_directory

def create_imagenet100_loaders(batch_size: int = 32, workers: int = 8, train_cfg = {}):
    """
    Returns (train_loader, val_loader) for ImageNet100.
    Automatically wraps in DistributedSampler if DDP is active.
    """
    if train_cfg and train_cfg.get('using_hpc',False):
        root_dir = Path('/work3/s234805/data/imagenet100/')
    else:
        root_dir = path_to_imagenet100()

    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    rank = (
        torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
    )

    # Instantiate datasets
    train_ds = ImageNet100(root_dir, train=True, validation=False)
    val_ds = ImageNet100(root_dir, train=False, validation=True)

    # Create samplers (for DDP) or None (for single-GPU)
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    per_gpu_bs = batch_size // world_size

    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_bs,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=per_gpu_bs,
        sampler=val_sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


class ImageNet20(ImageNet100):
    """A subset of ImageNet100 containing only 20 selected classes with reproducible train/val/test splits."""

    SELECTED_CLASSES = {
        "green mamba": True,
        "Doberman, Doberman pinscher": True,
        "cocktail shaker": True,
        "garter snake, grass snake": True,
        "pineapple, ananas": True,
        "American lobster, Northern lobster, Maine lobster, Homarus americanus": True,
        "ambulance": True,
        "cauliflower": True,
        "pirate, pirate ship": True,
        "safety pin": True,
        "theater curtain, theatre curtain": True,
        "red fox, Vulpes vulpes": True,
        "slide rule, slipstick": True,
        "walking stick, walkingstick, stick insect": True,
        "obelisk": True,
        "harmonica, mouth organ, harp, mouth harp": True,
        "mousetrap": True,
        "ski mask": True,
        "laptop, laptop computer": True,
        "gasmask, respirator, gas helmet": True,
    }

    def __init__(
        self,
        root_dir=None,
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    ):
        """
        Initialize ImageNet20 dataset with reproducible train/val/test splits.

        Args:
            root_dir: Root directory of the dataset
            split: Which split to load ('train', 'val', 'test')
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            random_seed: Random seed for reproducible splits (default: 42)
        """
        # Validate split ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of 'train', 'val', 'test'")

        # Initialize parent class with both train and validation data to get all data
        # We'll load from both original splits and then create our own splits
        self.split = split
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if root_dir is None:
            root_dir = load_imagenet100()

        self.root_dir = root_dir

        # Load both train and validation data from ImageNet100
        train_dir = os.path.join(self.root_dir, "train")
        val_dir = os.path.join(self.root_dir, "validation")

        train_dataset = load_from_disk(train_dir)
        val_dataset = load_from_disk(val_dir)

        # Combine both datasets
        from datasets import concatenate_datasets

        combined_dataset = concatenate_datasets([train_dataset, val_dataset])

        # Get the original label names
        original_label_names = combined_dataset.features["label"].names

        # Filter dataset to only include selected classes
        selected_indices = []
        original_labels_in_filtered = []

        for idx, item in enumerate(combined_dataset):
            class_name = original_label_names[item["label"]]
            if class_name in self.SELECTED_CLASSES:
                selected_indices.append(idx)
                if item["label"] not in original_labels_in_filtered:
                    original_labels_in_filtered.append(item["label"])

        # Create a new dataset with only selected classes
        filtered_dataset = combined_dataset.select(selected_indices)

        # Update number of classes and labels
        self.num_classes = len(self.SELECTED_CLASSES)
        self.labels = list(self.SELECTED_CLASSES.keys())

        # Create a mapping from original labels to new labels (0-19)
        self.label_mapping = {}
        for new_idx, class_name in enumerate(self.labels):
            original_idx = original_label_names.index(class_name)
            self.label_mapping[original_idx] = new_idx

        # Create reproducible train/val/test splits
        self.dataset = self._create_split(filtered_dataset, split)

        # Set transform states based on split
        self.is_train = split == "train"
        self.is_validation = split == "val"

        # Initialize transforms
        self.train_transforms = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transforms = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _create_split(self, dataset, split):
        """Create reproducible train/val/test splits."""
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Group indices by class to ensure balanced splits
        class_indices = {}
        for idx, item in enumerate(dataset):
            original_label = item["label"]
            if original_label not in class_indices:
                class_indices[original_label] = []
            class_indices[original_label].append(idx)

        # Create splits for each class separately
        split_indices = []

        for original_label, indices in class_indices.items():
            # Shuffle indices for this class
            indices = np.array(indices)
            np.random.shuffle(indices)

            n_samples = len(indices)
            n_train = int(n_samples * self.train_ratio)
            n_val = int(n_samples * self.val_ratio)
            n_test = n_samples - n_train - n_val  # Ensure all samples are used

            if split == "train":
                split_indices.extend(indices[:n_train])
            elif split == "val":
                split_indices.extend(indices[n_train : n_train + n_val])
            else:  # test
                split_indices.extend(indices[n_train + n_val :])

        # Sort indices to ensure consistent ordering
        split_indices.sort()

        return dataset.select(split_indices)

    def __getitem__(self, idx):
        """Fetches the sample at the given index and applies transforms."""
        item = self.dataset[idx]
        image = item["image"]
        original_label = item["label"]

        # Map the original label to the new label space
        label = self.label_mapping[original_label]

        # Apply transforms to the PIL image
        transforms_to_use = (
            self.train_transforms if self.is_train else self.val_transforms
        )
        if transforms_to_use:
            image = transforms_to_use(image.convert("RGB"))

        return image, label

    def get_by_index(self, idx, train=False):
        """Returns a sample dict given its index."""
        item = self.dataset[idx]
        image = item["image"]
        original_label = item["label"]
        label = self.label_mapping[original_label]

        # Apply transforms
        if train:
            tensor = self.__class__.transforms_train(image).unsqueeze(0)  # [1, C, H, W]
        else:
            tensor = self.__class__.transforms(image).unsqueeze(0)  # [1, C, H, W]
        return {"tensor": tensor, "label": label, "index": idx}

    def get_indices_from_class(self, class_idx, train=False, num_images=None):
        """Get indices of samples belonging to a specific class."""
        # Map the new class index back to the original class name
        class_name = self.labels[class_idx]
        original_class_idx = self.dataset.features["label"].names.index(class_name)

        indices = [
            i
            for i, item in enumerate(self.dataset)
            if item["label"] == original_class_idx
        ]
        if num_images is not None:
            indices = indices[:num_images]
        return indices

    def get_all_labels(self, train=False):
        """Returns a list of all labels for a given split."""
        return [self.label_mapping[label] for label in self.dataset["label"]]


def create_imagenet20_loaders(
    batch_size: int = 32,
    workers: int = 8,
    train_cfg={},
    include_test=False,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
):
    """
    Returns (train_loader, val_loader, test_loader) for ImageNet20 with reproducible splits.
    If include_test=False, returns (train_loader, val_loader).
    Automatically wraps in DistributedSampler if DDP is active.

    Args:
        batch_size: Total batch size across all GPUs
        workers: Number of data loading workers per GPU
        train_cfg: Training configuration dictionary
        include_test: Whether to include test loader in return
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_seed: Random seed for reproducible splits (default: 42)
    """
    if train_cfg and train_cfg.get("using_hpc", False):
        root_dir = Path("/work3/s234805/data/imagenet100/")
    else:
        root_dir = path_to_imagenet100()

    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )

    # Instantiate datasets with the same split configuration
    train_ds = ImageNet20(
        root_dir,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    val_ds = ImageNet20(
        root_dir,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    per_gpu_bs = batch_size // world_size

    # Create samplers (for DDP) or None (for single-GPU)
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_bs,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=per_gpu_bs,
        sampler=val_sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    if include_test:
        test_ds = ImageNet20(
            root_dir,
            split="test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        if world_size > 1:
            test_sampler = DistributedSampler(
                test_ds, num_replicas=world_size, rank=rank, shuffle=False
            )
        else:
            test_sampler = None

        test_loader = DataLoader(
            test_ds,
            batch_size=per_gpu_bs,
            sampler=test_sampler,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def get_imagenet20_split_info(
    root_dir=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
):
    """
    Get information about the ImageNet20 train/val/test splits.

    Returns:
        dict: Information about each split including sizes and class distributions
    """
    if root_dir is None:
        root_dir = path_to_imagenet100()

    # Create datasets for each split
    train_ds = ImageNet20(
        root_dir,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    val_ds = ImageNet20(
        root_dir,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    test_ds = ImageNet20(
        root_dir,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    def get_class_distribution(dataset):
        """Get the distribution of classes in a dataset."""
        class_counts = {}
        for item in dataset.dataset:
            original_label = item["label"]
            mapped_label = dataset.label_mapping[original_label]
            class_name = dataset.labels[mapped_label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

    info = {
        "train": {
            "size": len(train_ds),
            "class_distribution": get_class_distribution(train_ds),
        },
        "val": {
            "size": len(val_ds),
            "class_distribution": get_class_distribution(val_ds),
        },
        "test": {
            "size": len(test_ds),
            "class_distribution": get_class_distribution(test_ds),
        },
        "total_classes": len(ImageNet20.SELECTED_CLASSES),
        "class_names": list(ImageNet20.SELECTED_CLASSES.keys()),
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "random_seed": random_seed,
    }

    return info


# example use
if __name__ == "__main__":
    local_dataset_path = load_imagenet100()
    print(f"\nDataset setup process finished. Final dataset location: {local_dataset_path}")

    # Needed to locate the dataset directory
    data_directory = path_to_imagenet100()
    print(f"\nData directory for ImageNet100: {data_directory}")

    # Test the original ImageNet100
    train_dataset = ImageNet100(root_dir=data_directory, train=True)
    val_dataset = ImageNet100(root_dir=data_directory, validation=True)
    print(f"\nLength of training dataset object: {len(train_dataset)}")
    print(f"\nLength of validation dataset object: {len(val_dataset)}")

    train_loader, val_loader = create_imagenet100_loaders(batch_size=16, workers=4)
    print("\nImageNet100 DataLoaders work")

    # Test the new ImageNet20 with reproducible splits
    print("\n" + "=" * 60)
    print("Testing ImageNet20 with reproducible train/val/test splits")
    print("=" * 60)

    # Create individual datasets
    train_ds = ImageNet20(split="train", random_seed=42)
    val_ds = ImageNet20(split="val", random_seed=42)
    test_ds = ImageNet20(split="test", random_seed=42)

    # Create data loaders with test split
    train_loader, val_loader, test_loader = create_imagenet20_loaders(
        batch_size=32, include_test=True, random_seed=42
    )

    # Get split information
    split_info = get_imagenet20_split_info(random_seed=42)

    # Test reproducibility
    train_ds_20_repeat = ImageNet20(
        root_dir=data_directory, split="train", random_seed=42
    )
    val_ds_20_repeat = ImageNet20(root_dir=data_directory, split="val", random_seed=42)
    test_ds_20_repeat = ImageNet20(
        root_dir=data_directory, split="test", random_seed=42
    )

    print(f"\nReproducibility test (same seed=42):")
    print(f"Train sizes match: {len(train_ds_20) == len(train_ds_20_repeat)}")
    print(f"Val sizes match: {len(val_ds_20) == len(val_ds_20_repeat)}")
    print(f"Test sizes match: {len(test_ds_20) == len(test_ds_20_repeat)}")

    # Test different seed gives different splits
    train_ds_20_diff = ImageNet20(
        root_dir=data_directory, split="train", random_seed=123
    )
    print(f"\nDifferent seed test (seed=123):")
    print(f"Train size with different seed: {len(train_ds_20_diff)}")
    print(f"Same size with different seed: {len(train_ds_20) == len(train_ds_20_diff)}")

    # Get detailed split information
    split_info = get_imagenet20_split_info(data_directory)
    print(f"\nDetailed split information:")
    print(f"Classes: {split_info['total_classes']}")
    print(f"Train samples: {split_info['train']['size']}")
    print(f"Val samples: {split_info['val']['size']}")
    print(f"Test samples: {split_info['test']['size']}")
    print(f"Split ratios: {split_info['split_ratios']}")

    # Test some samples
    print(f"\nTesting sample access:")
    for i in range(2):
        sample = train_ds_20[i]
        print(f"Train sample {i}: Image shape: {sample[0].shape}, Label: {sample[1]}")

    for i in range(2):
        sample = val_ds_20[i]
        print(f"Val sample {i}: Image shape: {sample[0].shape}, Label: {sample[1]}")

    for i in range(2):
        sample = test_ds_20[i]
        print(f"Test sample {i}: Image shape: {sample[0].shape}, Label: {sample[1]}")
