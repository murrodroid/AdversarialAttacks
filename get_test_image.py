import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

def get_mnist_loader(batch_size=1, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root='./dataset', train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_cifar10_loader(batch_size=1, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.247,0.243,0.261))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./dataset', train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def denormalize_and_convert(tensor, dataset='mnist'):
    if dataset=='mnist':
        denorm = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))
        img = denorm(tensor).squeeze().numpy()
        img = np.clip(img, 0, 1)*255
        return img.astype(np.uint8)
    elif dataset=='cifar10':
        denorm = transforms.Normalize((-0.4914/0.247, -0.4822/0.243, -0.4465/0.261), (1/0.247, 1/0.243, 1/0.261))
        img = denorm(tensor)
        img = img.permute(1,2,0).numpy()
        img = np.clip(img, 0, 1)*255
        return img.astype(np.uint8)

def get_images_from_class(target_class, num_images=1, train=True, dataset='mnist'):
    loader = get_mnist_loader(train=train) if dataset=='mnist' else get_cifar10_loader(train=train)
    results = []
    for data, label in loader:
        if label.item() == target_class:
            img_tensor = data.squeeze(0)
            img_np = denormalize_and_convert(img_tensor, dataset)
            img_pil = Image.fromarray(img_np)
            results.append((data, img_pil))
            if len(results) >= num_images:
                break
    return results

def visualize_images(images, labels=None, save_path=None):
    n = len(images)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*2, rows*2))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray' if img.mode=='L' else None)
        if labels:
            plt.title(f"Class: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    plt.show()

def save_images(images, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    for i, img in enumerate(images):
        file_path = os.path.join(output_dir, f"{prefix}_{i}.png")
        img.save(file_path)
        file_paths.append(file_path)
        print(f"Image saved to {file_path}")
    return file_paths

def main():
    parser = argparse.ArgumentParser(description='Get images from a specific dataset')
    parser.add_argument('--class', type=int, required=True, dest='target_class', help='Target class')
    parser.add_argument('--count', type=int, default=1, help='Number of images to retrieve')
    parser.add_argument('--testset', action='store_true', help='Use test set instead of training set')
    parser.add_argument('--output', type=str, default='./images', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize the retrieved images')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','cifar10'], help='Dataset to use')
    args = parser.parse_args()
    print(f"Retrieving {args.count} images of class {args.target_class} from the {args.dataset} {'test' if args.testset else 'training'} set...")
    results = get_images_from_class(args.target_class, num_images=args.count, train=not args.testset, dataset=args.dataset)
    if not results:
        print(f"No images found for class {args.target_class}")
        return
    pil_images = [img for _, img in results]
    file_paths = save_images(pil_images, output_dir=args.output, prefix=f"{args.dataset}_class{args.target_class}")
    if args.visualize:
        visualize_images(pil_images, labels=[args.target_class]*len(pil_images), save_path=os.path.join(args.output, f"{args.dataset}_class{args.target_class}_visualization.png"))
    print(f"Successfully retrieved {len(file_paths)} images of class {args.target_class}")
    print(f"Use these images with your adversarial attack script:")
    print(f"python adversarial_attack.py --model path/to/model.pth --image {file_paths[0]} --target X")

if __name__=="__main__":
    main()