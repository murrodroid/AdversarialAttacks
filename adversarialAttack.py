import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

from src.datasets.cifar10 import Cifar10

from src.attacks.fsgm import fgsm_attack
from src.attacks.pgd import pgd_attack


class AdversarialAttacker:
    """
    A class for generating adversarial examples using various attack methods.
    """
    
    def __init__(self, model, dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the attacker with a model.
        
        Args:
            model: PyTorch model to attack
            device: Device to run the attack on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode
        
        if dataset == "mnist":
            self.dataset = Cifar10()
        elif dataset == "cifar10":
            self.dataset = Cifar10()
        else:
            raise ValueError("Dataset not supported")
    
    def preprocess_image(self, image):
        """
        Preprocess an image for the model.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        tensor = self.dataset.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def postprocess_image(self, tensor):
        """
        Convert tensor back to displayable image.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            PIL Image
        """
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
        """
        Run prediction on an image.
        
        Args:
            image: PIL Image or tensor
            
        Returns:
            Predicted class and confidence
        """
        if not torch.is_tensor(image):
            image = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(image)
            
        # Get prediction
        _, predicted = output.max(1)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence = probs[0][predicted.item()].item()
        
        return predicted.item(), confidence
          
    def deepfool_attack(self, image, target_class, max_iter=50, overshoot=0.02):
        """
        DeepFool attack implementation (adapted for targeted attacks).
        
        Args:
            image: Input image (PIL or tensor)
            target_class: Target class for attack
            max_iter: Maximum iterations
            overshoot: Parameter to increase the perturbation slightly
            
        Returns:
            Adversarial example as tensor
        """
        if not torch.is_tensor(image):
            image = self.preprocess_image(image)
        
        original_image = image.clone()
        adv_image = image.clone().detach().requires_grad_(True)
        
        output = self.model(adv_image)
        n_classes = output.shape[1]
        
        # Get original prediction
        _, orig_class = output.max(1)
        
        # If already target class, return original
        if orig_class.item() == target_class:
            return adv_image
            
        for i in range(max_iter):
            if adv_image.grad is not None:
                adv_image.grad.data.zero_()
                
            # Forward pass
            fs = self.model(adv_image)[0]
            f_target = fs[target_class]
            
            # Calculate gradients for all classes
            grads = []
            for k in range(n_classes):
                if k != target_class:
                    f_target.backward(retain_graph=True)
                    grad_target = adv_image.grad.data.clone()
                    adv_image.grad.data.zero_()
                    
                    fs[k].backward(retain_graph=True)
                    grad_k = adv_image.grad.data.clone()
                    adv_image.grad.data.zero_()
                    
                    # Gradient for target - other class
                    grads.append(grad_target - grad_k)
            
            # Find closest hyperplane
            with torch.no_grad():
                diffs = []
                for k in range(n_classes - 1):
                    diff = fs[target_class] - fs[k if k < target_class else k + 1]
                    diffs.append(abs(diff) / (grads[k].norm().item() + 1e-7))
                
                # Get closest non-target class
                l = np.argmin(diffs)
                
                # Compute perturbation
                grad = grads[l]
                diff = diffs[l]
                
                # Apply perturbation
                r_i = (diff + 1e-4) * grad / (grad.norm() + 1e-7)
                adv_image = adv_image + (1 + overshoot) * r_i
                
                # Project back to valid image range
                eta = torch.clamp(adv_image - original_image, -0.3, 0.3)
                adv_image = original_image + eta
                adv_image = torch.clamp(adv_image, (0 - 0.1307) / 0.3081, (1 - 0.1307) / 0.3081)
                
                adv_image = adv_image.detach().requires_grad_(True)
                
                # Check if attack is successful
                pred_class, _ = self.predict(adv_image)
                if pred_class == target_class:
                    print(f"Attack successful after {i+1} iterations!")
                    break
        
        return adv_image
    
    def carlini_wagner_attack(self, image, target_class, max_iter=1000, learning_rate=0.01, 
                              initial_const=0.01, binary_search_steps=5):
        """
        Carlini & Wagner L2 attack.
        
        Args:
            image: Input image (PIL or tensor)
            target_class: Target class for attack
            max_iter: Maximum iterations
            learning_rate: Learning rate for optimization
            initial_const: Initial value of the constant c
            binary_search_steps: Number of binary search steps to find optimal c
            
        Returns:
            Adversarial example as tensor
        """
        if not torch.is_tensor(image):
            image = self.preprocess_image(image)
        
        # Move to device
        image = image.to(self.device)
        
        # Change variable to work in tanh space
        def arctanh(x):
            return 0.5 * torch.log((1 + x) / (1 - x))
        
        # Box constraints in normalized space
        box_min = (0 - 0.1307) / 0.3081
        box_max = (1 - 0.1307) / 0.3081
        
        # Variables for binary search
        lower_bound = 0
        upper_bound = 1
        
        # Normalized image
        img_tanh = arctanh(torch.clamp((image - box_min) / (box_max - box_min) * 2 - 1, -0.999, 0.999))
        
        # The best attack found so far
        best_adv = None
        best_dist = float('inf')
        
        # Binary search for c
        c = initial_const
        for bs_step in range(binary_search_steps):
            # Initialize w
            w = nn.Parameter(torch.zeros_like(img_tanh), requires_grad=True)
            optimizer = torch.optim.Adam([w], lr=learning_rate)
            
            for i in range(max_iter):
                # Map w to image space
                # First, map to [-1, 1] using tanh
                w_tanh = torch.tanh(w + img_tanh)
                # Then map to [box_min, box_max]
                new_img = (w_tanh + 1) / 2 * (box_max - box_min) + box_min
                
                # Calculate output
                output = self.model(new_img)
                
                # Calculate the distance (L2)
                l2_dist = torch.sum((new_img - image) ** 2)
                
                # Calculate objective function
                target_onehot = torch.zeros(output.shape, device=self.device)
                target_onehot[0, target_class] = 1
                
                # Using CW loss: max(max{Z(x')_i: i≠t} - Z(x')_t, -κ)
                real = torch.sum(target_onehot * output)
                other = torch.max((1 - target_onehot) * output)
                zero = torch.tensor(0.0).to(self.device)
                
                # For targeted attack, we want to minimize this
                f_loss = torch.max(other - real, zero)
                
                # Total loss
                loss = l2_dist + c * f_loss
                
                # Gradient update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Check success and update best result
                pred_class = output.argmax(1).item()
                if pred_class == target_class:
                    curr_dist = l2_dist.item()
                    if curr_dist < best_dist:
                        best_dist = curr_dist
                        best_adv = new_img.detach()
                        
                # Print progress
                if (i + 1) % 100 == 0:
                    print(f"Step {bs_step+1}/{binary_search_steps}, Iter {i+1}/{max_iter}, "
                          f"Loss: {loss.item():.4f}, Dist: {l2_dist.item():.4f}, "
                          f"Class: {pred_class}")
            
            # Update c based on binary search
            if best_adv is not None:
                upper_bound = c
                c = (lower_bound + upper_bound) / 2
            else:
                lower_bound = c
                c = c * 10 if upper_bound == 1 else (lower_bound + upper_bound) / 2
                
        return best_adv if best_adv is not None else new_img.detach()
    
    def generate_adversarial(self, image, target_class, attack_type='fgsm', **kwargs):
        """
        Generate an adversarial example using the specified attack.
        
        Args:
            image: Original image (PIL or numpy array)
            target_class: Target class for the attack
            attack_type: Type of attack ('fgsm', 'pgd', 'deepfool', 'cw')
            **kwargs: Additional parameters for the specific attack
            
        Returns:
            Tuple of (original_image, adversarial_image, original_class, adversarial_class)
        """
        tensor_image = self.preprocess_image(image) if not torch.is_tensor(image) else image.clone()
        tensor_image = tensor_image.to(self.device)
        
        # Get original prediction
        orig_class, orig_conf = self.predict(tensor_image)
        print(f"Original image classified as {orig_class} with {orig_conf:.4f} confidence")
        
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
        
        print(f"creating attack image using {attack_type}") 
        
        # Get adversarial prediction
        adv_class, adv_conf = self.predict(adv_tensor)
        print(f"Adversarial image classified as {adv_class} with {adv_conf:.4f} confidence")
        
        # Convert tensors back to images
        orig_img = self.postprocess_image(tensor_image)
        adv_img = self.postprocess_image(adv_tensor)
        
        return orig_img, adv_img, orig_class, adv_class


def visualize_results(original_image, adversarial_image, orig_class, adv_class, save_path=None):
    """
    Visualize the original and adversarial images with predictions.
    
    Args:
        original_image: Original PIL image
        adversarial_image: Adversarial PIL image
        orig_class: Original prediction class
        adv_class: Adversarial prediction class
        save_path: Path to save the visualization
    """
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


def load_model(model_path, dataset):
    """
    Load a PyTorch model from a specified path.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded PyTorch model
    """
    try:
        if dataset == "mnist":
            from mnist import Net
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net().to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True))
        elif dataset == "cifar10":
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        else:
            raise ValueError("Dataset not supported")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial examples for MNIST')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','cifar10'])
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--target', type=int, required=True, help='Target class (0-9)')
    parser.add_argument('--attack', type=str, default='fgsm', 
                        choices=['fgsm', 'pgd', 'deepfool', 'cw'],
                        help='Attack type')
    parser.add_argument('--epsilon', type=float, default=0.5, help='Epsilon for FGSM/PGD attack')
    parser.add_argument('--alpha', type=float, default=0.05, help='Step size for PGD attack')
    parser.add_argument('--iterations', type=int, default=200, help='Max iterations')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Check if target class is valid
    if not 0 <= args.target <= 9:
        print("Error: Target class must be between 0 and 9 for MNIST")
        return
    
    # Load the model
    model = load_model(args.model, args.dataset)
    
    # Create the attacker
    attacker = AdversarialAttacker(model, args.dataset)

    images = attacker.dataset.get_sample_from_class(1)
    image = images[0]
       
    # try:
    #     image = Image.open(args.image)
    #     if args.dataset == "mnist":
    #         image = image.convert('L').resize((28,28))
    #     elif args.dataset == "cifar10":
    #         image = image.convert('RGB').resize((32,32))
    # except Exception as e:
    #     print(f"Error loading image: {e}")
    #     return
    
    # Create attack kwargs from args
    attack_kwargs = {
        'epsilon': args.epsilon,
        # 'max_iter': args.iterations
    }
    if args.attack == 'pgd':
        attack_kwargs['alpha'] = args.alpha
    
    # Generate adversarial example
    orig_img, adv_img, orig_class, adv_class = attacker.generate_adversarial(
        image, args.target, args.attack, **attack_kwargs
    )
    
    # Create output directory if needed
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
    
    # Visualize results
    visualize_results(orig_img, adv_img, orig_class, adv_class, viz_path)


if __name__ == "__main__":
    main()