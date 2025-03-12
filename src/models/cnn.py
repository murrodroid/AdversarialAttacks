import torch.nn as nn

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
      nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
      nn.MaxPool2d(2, 2)
    )
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512 * 4 * 4, 512), nn.ReLU(),
      nn.Linear(512, 100)
    )

  def forward(self, x):
    return self.fc(self.conv(x))