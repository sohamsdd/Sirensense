import torch
from torchvision import models, transforms
from torchcam.methods import GradCAM
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# Load model
# -----------------------------
model = models.resnet18(pretrained=True)
model.eval()

# -----------------------------
# GradCAM setup
# -----------------------------
cam_extractor = GradCAM(model, target_layer='layer4')

# -----------------------------
# Image preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Folder setup
# -----------------------------
input_folder = "Spectrograms_Gradcam"
output_folder = "GradCAM_Results"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Process each image
# -----------------------------
for img_name in os.listdir(input_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, img_name)
    print(f"Processing: {img_path}")

    # Load image
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)

    # Forward pass
    out = model(input_tensor)
    class_idx = out.squeeze().argmax().item()

    # GradCAM
    activation_maps = cam_extractor(class_idx, out)

    # -----------------------------
    # Fix: GradCAM returns a list
    # -----------------------------
    cam_tensor = activation_maps[0]  # Take the first map
    cam = cam_tensor.squeeze().detach().cpu().numpy()

    # Normalize to [0,1]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Resize heatmap to original image size
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, resample=Image.BILINEAR))

    # Apply colormap
    heatmap_color = plt.get_cmap('jet')(cam_resized / 255.0)[:, :, :3]
    heatmap_overlay = (0.4 * heatmap_color + 0.6 * np.array(img) / 255.0)
    heatmap_overlay = np.clip(heatmap_overlay, 0, 1)

    # Save combined visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_overlay)
    plt.title('Grad-CAM Overlay')
    plt.axis('off')

    save_path = os.path.join(output_folder, f"gradcam_{img_name}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Saved Grad-CAM visualization to: {save_path}")