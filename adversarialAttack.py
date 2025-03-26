import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import random

from src.datasets.cifar10 import Cifar10

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.utils.torch_util import getDevice

class AdversarialAttacker:
    def __init__(self, model, dataset, device = getDevice()):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        if dataset == "mnist":
            self.dataset = Cifar10()
        elif dataset == "cifar10":
            self.dataset = Cifar10()
        else:
            raise ValueError("Dataset not supported")
    
    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        tensor = self.dataset.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def postprocess_image(self, tensor):
        tensor = tensor.squeeze(0).detach().cpu()
        tensor = self.dataset.inverse_transform(tensor)
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL image
        img_np = tensor.permute(1, 2, 0).numpy()
        if img_np.shape[2] == 1:  # If grayscale, remove channel dimension
            img_np = img_np.squeeze(2)
        
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        return img
    
    def predict(self, image):
        if not torch.is_tensor(image):
            image = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(image)
            
        _, predicted = output.max(1)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence = probs[0][predicted.item()].item()
        
        return predicted.item(), confidence
    
    def generate_adversarial(self, image, target_class, attack_type='fgsm', **kwargs):
        tensor_image = self.preprocess_image(image) if not torch.is_tensor(image) else image.clone()
        tensor_image = tensor_image.to(self.device)
        
        orig_class, orig_conf = self.predict(tensor_image)
        print(f"Original image classified as {self.dataset.get_class_name(orig_class)} ({orig_class}) with {orig_conf:.4f} confidence")
        
        # Generate adversarial example
        if attack_type.lower() == 'fgsm':
            adv_tensor = fgsm_attack(self.model, tensor_image, target_class, **kwargs)
        elif attack_type.lower() == 'pgd':
            adv_tensor = pgd_attack(self.model, tensor_image, target_class, **kwargs)
        elif attack_type.lower() == 'deepfool':
            adv_tensor = self.deepfool_attack(tensor_image, target_class, **kwargs)
        elif attack_type.lower() in ['cw', 'carlini_wagner']:
            adv_tensor = self.carlini_wagner_attack(tensor_image, target_class, **kwargs)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        print(f"creating attack image using {attack_type} targeting class {target_class}") 
        
        adv_class, adv_conf = self.predict(adv_tensor)
        print(f"Adversarial image classified as {self.dataset.get_class_name(adv_class)} ({adv_class}) with {adv_conf:.4f} confidence")
        
        orig_img = self.postprocess_image(tensor_image)
        adv_img = self.postprocess_image(adv_tensor)
        
        return orig_img, adv_img, orig_class, adv_class


def visualize_results(original_image, adversarial_image, orig_class, adv_class, save_path=None):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original: {orig_class}")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f"Adversarial: {adv_class}")
    plt.imshow(adversarial_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    diff = np.array(adversarial_image) - np.array(original_image)
    plt.title("Difference")
    plt.imshow(diff, cmap='bwr')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def load_model(dataset):
    if dataset == "cifar10":
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    else:
        raise ValueError("Dataset not supported")
    return model


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser(description='Generate adversarial examples for different datasets')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist','cifar10'])
    parser.add_argument('--source', type=int, required=True, help='Source class (0-9)')
    parser.add_argument('--target', type=int, required=True, help='Target class (0-9)')
    parser.add_argument('--attack', type=str, default='fgsm', 
                        choices=['fgsm', 'pgd', 'deepfool', 'cw'],
                        help='Attack type')
    parser.add_argument('--epsilon', type=float, default=0.03, help='Epsilon for FGSM/PGD attack')
    parser.add_argument('--alpha', type=float, default=0.05, help='Step size for PGD attack')
    parser.add_argument('--iterations', type=int, default=200, help='Max iterations')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    model = load_model(args.dataset)
    
    attacker = AdversarialAttacker(model, args.dataset)

    image = attacker.dataset.get_sample_from_class(args.source)[0]
    attack_kwargs = { 'epsilon': args.epsilon, # 'max_iter': args.iterations
    }
    if args.attack == 'pgd':
        attack_kwargs['alpha'] = args.alpha
    
    orig_img, adv_img, orig_class, adv_class = attacker.generate_adversarial(
        image, args.target, args.attack, **attack_kwargs
    )
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        orig_path = os.path.join(args.output, 'original.png')
        adv_path = os.path.join(args.output, 'adversarial.png')
        viz_path = os.path.join(args.output, 'visualization.png')
        
        orig_img.save(orig_path)
        adv_img.save(adv_path)
        
        print(f"Original image saved to {orig_path}")
        print(f"Adversarial image saved to {adv_path}")
    else:
        viz_path = None
    
    visualize_results(orig_img, adv_img, orig_class, adv_class, viz_path)


if __name__ == "__main__":
    main()