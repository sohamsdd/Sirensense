import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import cv2

# -------------------------------
# CNN Model (same as train.py)
# -------------------------------
class SpectrogramClassifierCNN(pl.LightningModule):
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
# Model Loading
# -------------------------------
def load_model(checkpoint_path="spectogram_classifier.ckpt", num_classes=4):
    try:
        model = SpectrogramClassifierCNN.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        return None


# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor, np.array(image)


# -------------------------------
# Grad-CAM Implementation
# -------------------------------
def generate_gradcam(model, input_tensor, target_class=None):
    # Hook the feature maps and gradients of the last conv layer
    conv_layer = model.features[-3]  # Last Conv2d layer (before BatchNorm2d(512))
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Register hooks
    forward_handle = conv_layer.register_forward_hook(forward_hook)
    backward_handle = conv_layer.register_backward_hook(backward_hook)


    # Forward pass
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero grads and backward
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    # Compute Grad-CAM
    grads = gradients[0]        # [batch, channels, h, w]
    acts = activations[0]       # [batch, channels, h, w]
    weights = grads.mean(dim=(2, 3), keepdim=True)  # Global average pooling
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Normalize and resize to input size
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (512, 128))

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return cam


# -------------------------------
# Prediction + Visualization
# -------------------------------
def predict_and_visualize(model, image_paths, class_names=None, save_dir="gradcam_outputs"):
    os.makedirs(save_dir, exist_ok=True)

    if class_names is None:
        class_names = ["emergency", "emergency2urban", "urban", "urban2emergency"]

    for img_path in image_paths:
        input_tensor, orig_img = preprocess_image(img_path)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            pred_idx = np.argmax(probs)
            pred_class = class_names[pred_idx]

        # Generate Grad-CAM
        cam = generate_gradcam(model, input_tensor, target_class=pred_idx)

        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 🔧 Resize original image to match CAM size
        orig_resized = cv2.resize(orig_img, (heatmap.shape[1], heatmap.shape[0]))

        # Overlay heatmap on top of resized input
        overlay = (0.4 * heatmap + 0.6 * orig_resized).astype(np.uint8)


        # Plot Input vs Grad-CAM
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title("Input Spectrogram")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f"Grad-CAM ({pred_class})")
        plt.axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f" {img_path} → Predicted: {pred_class}")
        print("Probabilities:")
        for cls, p in zip(class_names, probs):
            print(f"  {cls:20s}: {p*100:.2f}%")
        print(f"Grad-CAM saved to: {save_path}\n")


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    checkpoint_path = "spectogram_classifier.ckpt"
    model = load_model(checkpoint_path, num_classes=4)

    test_images = [

        "Spectrograms/emergency/emergency33.png",
        "Spectrograms/emergency/emergency34.png",
        "Spectrograms/emergency/emergency35.png",
        "Spectrograms/emergency/emergency36.png",
        "Spectrograms/emergency/emergency37.png",
        "Spectrograms/emergency/emergency38.png",

    ]

    if model:
        predict_and_visualize(model, test_images)
