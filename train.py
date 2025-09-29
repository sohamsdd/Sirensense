import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# -------------------------------
# CNN Model (Improved Architecture)
# -------------------------------
class SpectrogramClassifierCNN(pl.LightningModule):
    """
    A deeper, more robust CNN model for spectrogram classification.
    Incorporates additional convolutional layers and Batch Normalization
    to improve training stability and performance.
    """
    def __init__(self, num_classes=4, lr=1e-4):
        super(SpectrogramClassifierCNN, self).__init__()
        self.save_hyperparameters()

        # Define the convolutional blocks
        self.features = nn.Sequential(
            # Block 1: 3x128x512 -> 32x64x256
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32x64x256 -> 64x32x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64x32x128 -> 128x16x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 128x16x64 -> 256x8x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 256x8x32 -> 512x4x16
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Define the fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.6), # Increased dropout for better generalization
            nn.Linear(512 * 4 * 16, 256), # Output size from last pooling layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )
        
        self.lr = lr

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten the tensor
        x = self.classifier(x)
        return x # Returns logits, which F.cross_entropy expects

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# -------------------------------
# Data Preparation
# -------------------------------
def get_dataloaders(data_dir="Spectrograms", batch_size=32):
    """
    Loads and splits the dataset into training and validation sets.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 512)), # Resize to a standard size
        transforms.ToTensor(),
    ])

    # Load dataset from the directory
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Split the dataset (80/20)
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, random_state=42, stratify=dataset.targets
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # ⚠️ num_workers=0 is safest on Windows. Adjust for speedup on other OS.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


# -------------------------------
# Training Script
# -------------------------------
if __name__ == "__main__":
    data_dir = "Spectrograms"
    batch_size = 32
    max_epochs = 20 # Increased epochs for better training
    
    if not os.path.exists(data_dir):
        print(f"ERROR: The '{data_dir}' directory was not found. Please create it and place your class folders inside.")
    else:
        train_loader, val_loader = get_dataloaders(data_dir, batch_size)
        
        # Initialize the new model
        model = SpectrogramClassifierCNN(num_classes=4, lr=1e-4)

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="cpu", # Change to "gpu" if CUDA is available
            devices=1,
            log_every_n_steps=10
        )

        trainer.fit(model, train_loader, val_loader)

        # Save final model checkpoint
        ckpt_path = "spectogram_classifier.ckpt"
        trainer.save_checkpoint(ckpt_path)
        print(f"✅ Training complete! Model saved at {ckpt_path}")
