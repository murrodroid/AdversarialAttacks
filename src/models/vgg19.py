import torch
import torch.nn as nn
from torchvision.models import vgg19,VGG19_Weights

class VGG19:
    def __init__(self,num_classes:int = 1000): 
        self.num_classes = num_classes
    
    def load(self):
        model = vgg19(weights=VGG19_Weights.DEFAULT).eval()
        for p in model.parameters():
            p.requires_grad = False # Freeze the model parameters
        model.classifier[-1] = nn.Linear(4096,self.num_classes) # Change the last layer to match the number of classes
        model.eval() # Set the model to evaluation mode
        return model
