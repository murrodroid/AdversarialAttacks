import torch
import os
import shutil 
from datasets import load_from_disk,load_dataset,DatasetDict
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as T


# Downloading imagenet100 dataset from huggingface
# saving it locally then to a folder in the git repository and deleting the temporary cache folder
def load_imagenet100():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_data_dir = os.path.join(script_dir, '..', 'data')
    final_dataset_path = os.path.join(project_data_dir, 'imagenet100')

    if os.path.exists(final_dataset_path):
        print("ImageNet100 dataset already exists in the expected location. Skipping download.")
        return final_dataset_path
    
    os.makedirs(project_data_dir, exist_ok=True)
    print(f"Ensured project data directory exists at: {project_data_dir}")

    temp_cache_dir = "C:\\cache_imagenet100" 
    os.makedirs(temp_cache_dir, exist_ok=True)
    print(f"Ensured cache directory exists at: {project_data_dir}")

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


class ImageNet100(Dataset):
    def __init__(self, root_dir, train = False,validation = False):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (bool): If True, load the training set.
            validation (bool): If True, load the validation set.
        """
        self.root_dir = root_dir

        split_name = 'train' if train else 'validation'
        self.dire = os.path.join(self.root_dir, split_name)
        self.dataset = load_from_disk(self.dire)

        # self.transforms = transforms
        self.transforms =  T.Compose([
                                T.RandomResizedCrop(224),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]) if train else T.Compose([
                                                    T.Resize(256),
                                                    T.CenterCrop(224),
                                                    T.ToTensor(),
                                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])


        self.num_classes = self.dataset.features['label'].num_classes


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
        image = item['image']
        label = item['label']

        # 6. Apply transforms to the PIL image.
        if self.transforms:
            # Ensure image is RGB, as required by standard models
            image = self.transforms(image.convert('RGB'))
            
        return image, torch.tensor(label, dtype=torch.long)


def create_imagenet100_loaders(root_dir: str,
                               batch_size: int,
                               workers: int = 8):
    """
    Returns (train_loader, val_loader) for ImageNet100.
    Automatically wraps in DistributedSampler if DDP is active.
    """
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
    train_ds = ImageNet100(root_dir, train=True, validation=False)
    val_ds = ImageNet100(root_dir, train=False, validation=True)

    # Create samplers (for DDP) or None (for single-GPU)
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                           rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                         rank=rank, shuffle=False)
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


def path_to_imagenet100():
    """
    Finds the ImageNet100 dataset in the expected location.
    Returns the path to the dataset directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir)) 
    data_directory = os.path.join(project_root, 'src', 'data', 'imagenet100')

    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"ImageNet100 dataset not found at {data_directory}")

    return data_directory

# example use
if __name__ == "__main__":
    local_dataset_path = load_imagenet100()
    print(f"\nDataset setup process finished. Final dataset location: {local_dataset_path}")

    # Needed to locate the dataset directory   
    data_directory = path_to_imagenet100()
    print(f"\nData directory for ImageNet100: {data_directory}")

    train_dataset = ImageNet100(root_dir=data_directory,train=True)
    val_dataset = ImageNet100(root_dir=data_directory, validation=True)
    print(f"\nLength of training dataset object: {len(train_dataset)}")
    print(f"\nLength of validation dataset object: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print("\nDataLoaders works")

    # We want to test the getitem method for two samples
    for i in range(2):
        img, label = train_dataset[i]
        print(f"Sample {i} from train dataset: Image shape: {img.shape}, Label: {label.item()}")
    for i in range(2):
        img, label = val_dataset[i]
        print(f"Sample {i} from test dataset: Image shape: {img.shape}, Label: {label.item()}")

