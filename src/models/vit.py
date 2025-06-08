import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import load_from_disk
from timm import create_model
from tqdm import tqdm
from src.datasets.imagenet import load_imagenet100

class ApplyTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch):
        batch['pixel_values'] = [self.transforms(img.convert('RGB')) for img in batch['image']]
        del batch['image'] 
        return batch

class ViTClassifier:
    """
    A complete class to handle ViT model training and evaluation on ImageNet-100.
    """
    def __init__(self, data_root, model_name='vit_base_patch16_224', batch_size=32, lr=3e-5, epochs=5):
        """
        Initializes the classifier, datasets, model, and training components.

        Args:
            data_root (str): The root directory of the dataset (e.g., '.../data/imagenet100').
            model_name (str): The name of the model to use from the 'timm' library.
            batch_size (int): The batch size for training and evaluation.
            lr (float): The learning rate for the optimizer.
            epochs (int): The total number of training epochs.
        """
        self.data_root = data_root
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        
        # Setup device, prioritizing GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")

        # Load the dataset if not already done
        if not os.path.exists(os.path.join(self.data_root, 'train')) or not os.path.exists(os.path.join(self.data_root, 'validation')):
            print(f"Dataset not found. Downloading ImageNet-100...")
            directory = load_imagenet100()
        else:
            print(f"Using existing dataset at {self.data_root}")

        # Setup datasets and dataloaders
        self._setup_datasets()

        # Setup model, optimizer, and loss function
        self._setup_model()

    def _setup_datasets(self):
        """Loads train/validation sets and creates DataLoader instances."""
        print("--- Setting up datasets ---")
        
        # Load datasets
        train_path = os.path.join(self.data_root, 'train')
        val_path = os.path.join(self.data_root, 'validation')
        
        self.train_dataset = load_from_disk(train_path)
        self.val_dataset = load_from_disk(val_path)
        
        # Number of classes from the dataset
        self.num_classes = self.train_dataset.features['label'].num_classes
        print(f"Dataset has {self.num_classes} classes.")

        # Define transforms with randomized cropping and normalization
        train_transforms = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset.set_transform(ApplyTransforms(train_transforms))
        self.val_dataset.set_transform(ApplyTransforms(val_transforms))


        num_workers = 4 if self.device.type == 'cuda' else 0 
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
    def _setup_model(self):
        """Initializes the ViT model, optimizer, and criterion."""
        print(f"--- Setting up model: {self.model_name} ---")
        
        # Load a pretrained Vision Transformer model and replace the final layer
        self.model = create_model(
            self.model_name,
            pretrained=True,
            num_classes=self.num_classes
        ).to(self.device)
        
        # Setup optimizer and loss function
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)


    def _train_one_epoch(self):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # The collate_fn of DataLoader now stacks the tensors for you
            inputs = batch['pixel_values'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def evaluate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(self.val_loader, desc="Evaluating", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                inputs = torch.stack(batch['pixel_values']).to(self.device)
                labels = torch.tensor(batch['label']).to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def train(self):
        """The main training loop."""
        print("\n--- Starting Training ---")
        for epoch in range(self.epochs):
            avg_train_loss = self._train_one_epoch()
            avg_val_loss, val_accuracy = self.evaluate()
            
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Accuracy: {val_accuracy:.4f}")
        print("--- Training Finished ---")

# The main block should be protected to work correctly with multiprocessing
if __name__ == '__main__':
    # Needed for finding the dataset path:
    # Note: __file__ might not be defined in all environments (e.g., interactive notebooks)
    # Using a hardcoded or relative path might be more stable if issues arise.

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir)) 
    data_directory = os.path.join(project_root,'fagprojekt','AdversarialAttacks','src', 'data', 'imagenet100')


    # Create and run the classifier
    classifier = ViTClassifier(
        data_root=data_directory,
        batch_size=32,  # Adjust based on your GPU memory
        epochs=1,       # Set to a lower number for quick testing
    )
    classifier.train()