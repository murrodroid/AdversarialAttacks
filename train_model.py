import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm, trange

from src.models.cnn import CNN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def train(epoch = 20):
    trainset = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in trange(epoch):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs),labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

def test():
    testset = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


if __name__ == "__main__":
    SHOULD_TRAIN = True
    MODEL_PATH = "./model_weights/cifar_100_checkpoint.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    model = CNN().to(device)

    if SHOULD_TRAIN:
        train()
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    acc = test()
    print(f"Test Accuracy: {acc:.4f}")