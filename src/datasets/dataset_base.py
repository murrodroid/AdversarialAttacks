from abc import ABC
import torch

class DatasetBase(ABC):
  def __init__(self, dataset, path='./data', download=True):
    self.download = download
    self.path = path
    self._dataset = dataset

  @property
  def transforms(self):
    return self.transform

  @property
  def inverse_transforms(self):
    return self.inverse_transform
  
  def get_class_name(self, class_id):
    return self.labels[class_id]

  def get_sample_from_class(self, class_idx, train=True, num_images=1):
    dataset = self._dataset(root=self.path, train=train, transform=self.transforms, download=self.download)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    results = []
    for data, label in loader:
        if label.item() == class_idx:
            results.append(data)
            if len(results) >= num_images:
                break
    return results
