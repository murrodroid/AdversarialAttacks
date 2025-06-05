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






if __name__ == "__main__":
    local_dataset_path = load_imagenet100()
    print(f"\nDataset setup process finished. Final dataset location: {local_dataset_path}")

