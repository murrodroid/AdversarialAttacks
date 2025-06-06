from abc import ABC
import torch
import random
from torchvision import transforms

class DatasetBase(ABC):

    def __init__(self, dataset, path="./data", download=True):
        self.download = download
        self.path = path
        self._dataset = dataset

    @staticmethod
    def transforms(image):
        return None

    @staticmethod
    def inverse_transforms(tensor):
        return None

    def get_class_name(self, class_id):
        return self.labels[class_id]

    def get_sample_from_class(self, class_idx, train=True, num_images=1):
        try:
            if self.__class__.transforms == DatasetBase.transforms:
                transform_func = transforms.ToTensor()
            else:
                transform_func = lambda img: self.__class__.transforms(img)

            dataset = self._dataset(
                root=self.path,
                train=train,
                transform=transform_func,
                download=self.download,
            )
            indices = [
                i for i, label in enumerate(dataset.targets) if label == class_idx
            ]

            num_to_sample = min(num_images, len(indices))
            selected_indices = random.sample(indices, num_to_sample)

            results = []
            for idx in selected_indices:
                try:
                    data, label = dataset[idx]
                    sample_dict = {"index": idx, "tensor": data.unsqueeze(0)}
                    results.append(sample_dict)

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
