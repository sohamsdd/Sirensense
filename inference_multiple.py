import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

# -------------------------------
# CNN Model (must be the same as the training script)
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

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(512 * 4 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )
        
        self.lr = lr

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------------------
# Inference Pipeline
# -------------------------------
def load_model(checkpoint_path="spectrogram_classifier.ckpt", num_classes=4):
    """
    Loads a trained model from a checkpoint.
    """
    try:
        model = SpectrogramClassifierCNN.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
        model.eval() # Set model to evaluation mode
        return model
    except FileNotFoundError:
        print(f"Error: Checkpoint file '{checkpoint_path}' not found. Please train the model first.")
        return None

def preprocess_image(image_path):
    """
    Preprocesses a single image for model inference.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
    ])
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0) # Add a batch dimension
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return None

def predict_multiple(model, image_paths, class_names=None):
    """
    Performs inference on a list of image paths and returns the predictions.
    """
    if model is None:
        return {}

    predictions = {}
    for image_path in image_paths:
        image_tensor = preprocess_image(image_path)
        if image_tensor is not None:
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

                if class_names is None:
                    class_names = ["emergency", "emergency2urban", "urban", "urban2emergency"]
                
                # Convert to percentages
                probs_percent = [float(p) * 100 for p in probs]
                
                # Pair class with probability
                results = {cls: round(p, 4) for cls, p in zip(class_names, probs_percent)}

                predicted_idx = probs.argmax()
                predicted_class = class_names[predicted_idx]
                
                predictions[image_path] = {
                    "predicted_class": predicted_class,
                    "probabilities": results
                }
    return predictions

# -------------------------------
# Run Example
# -------------------------------
if __name__ == "__main__":
    checkpoint_path = "spectogram_classifier.ckpt"
    
    # List of image paths to test
    test_images = [

        "Spectrograms/urban/urban41.png",
        "Spectrograms/urban/urban42.png",
        "Spectrograms/urban/urban43.png",


        
    ]

    model = load_model(checkpoint_path, num_classes=4)
    if model:
        all_predictions = predict_multiple(model, test_images)
        
        if all_predictions:
            print("✅ Inference complete for multiple files:")
            for path, result in all_predictions.items():
                print(f"\nFile: {path}")
                print(f"Predicted class: {result['predicted_class']}")
                print("Class probabilities:", result['probabilities'])
        else:
            print("No valid predictions could be made. Check file paths and model checkpoint.")

# "Spectrograms/emergency2urban/emergency2urban20.png"