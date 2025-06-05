import torchvision.transforms as transforms
from datasets import load_dataset, DatasetDict
import os
import shutil 
from .dataset_base import DatasetBase

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




class ImageNet100(DatasetBase): 
    def __init__(self, dataset_path='../data/imagenet100', split='train'): # dataset_path is now mandatory
        self.dataset_root_path = dataset_path
        self.split = split
        
        try:
            full_ds_dict = DatasetDict.load_from_disk(self.dataset_root_path)
            self.dataset = full_ds_dict[self.split]
            print(f"Successfully loaded '{self.split}' split from {self.dataset_root_path}")
        except Exception as e:
            print(f"ERROR!: Make sure the dataset is downloaded using load_imagenet100 function and available at {self.dataset_root_path}")
            self.dataset = None # Or raise error

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        inv_mean = [-m/s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        inv_std = [1/s for s in (0.229, 0.224, 0.225)]
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(tuple(inv_mean), tuple(inv_std))
        ])

        # Retrieve label names from the dataset
        self.labels = []
        if self.dataset is not None and "label" in self.dataset.features:
            label_info = self.dataset.features["label"]
            if hasattr(label_info, "names"):
                # ClassLabel type: use .names
                self.labels = label_info.names
            else:
                # Otherwise, fetch unique values (numeric or string)
                self.labels = list(set(self.dataset["label"]))


if __name__ == "__main__":
    local_dataset_path = load_imagenet100()
    print(f"\nDataset setup process finished. Final dataset location: {local_dataset_path}")

