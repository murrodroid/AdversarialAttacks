import torch
import os
import shutil
import tempfile
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
    """A subset of ImageNet100 containing only 20 selected classes."""

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
        "Dutch oven": True,
        "gasmask, respirator, gas helmet": True,
    }

    def __init__(self, root_dir=None, train=False, validation=False):
        super().__init__(root_dir, train, validation)

        # Get the original label names
        original_label_names = self.dataset.features["label"].names

        # Filter dataset to only include selected classes
        selected_indices = []
        original_labels_in_filtered = []

        for idx, item in enumerate(self.dataset):
            class_name = original_label_names[item["label"]]
            if class_name in self.SELECTED_CLASSES:
                selected_indices.append(idx)
                if item["label"] not in original_labels_in_filtered:
                    original_labels_in_filtered.append(item["label"])

        # Create a new dataset with only selected classes
        self.dataset = self.dataset.select(selected_indices)

        # Update number of classes and labels
        self.num_classes = len(self.SELECTED_CLASSES)
        self.labels = list(self.SELECTED_CLASSES.keys())

        # Create a mapping from original labels to new labels (0-19)
        self.label_mapping = {}
        for new_idx, class_name in enumerate(self.labels):
            original_idx = original_label_names.index(class_name)
            self.label_mapping[original_idx] = new_idx

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


def create_imagenet20_loaders(batch_size: int = 32, workers: int = 8, train_cfg={}):
    """
    Returns (train_loader, val_loader) for ImageNet20.
    Automatically wraps in DistributedSampler if DDP is active.
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

    # Instantiate datasets
    train_ds = ImageNet20(root_dir, train=True, validation=False)
    val_ds = ImageNet20(root_dir, train=False, validation=True)

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


# example use
if __name__ == "__main__":
    local_dataset_path = load_imagenet100()
    print(f"\nDataset setup process finished. Final dataset location: {local_dataset_path}")

    # Needed to locate the dataset directory
    data_directory = path_to_imagenet100()
    print(f"\nData directory for ImageNet100: {data_directory}")

    train_dataset = ImageNet100(root_dir=data_directory, train=True)
    val_dataset = ImageNet100(root_dir=data_directory, validation=True)
    print(f"\nLength of training dataset object: {len(train_dataset)}")
    print(f"\nLength of validation dataset object: {len(val_dataset)}")

    train_loader,val_loader = create_imagenet100_loaders(batch_size=16,workers=4)
    print("\nDataLoaders works")

    # We want to test the getitem method for two samples
    for i in range(2):
        sample = train_dataset[i]
        print(
            f"Sample {i} from train dataset: Image shape: {sample['tensor'].shape}, Label: {sample['label']}"
        )
    for i in range(2):
        sample = val_dataset[i]
        print(
            f"Sample {i} from test dataset: Image shape: {sample['tensor'].shape}, Label: {sample['label']}"
        )
