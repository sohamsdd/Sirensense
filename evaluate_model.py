import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from train import SpectogramClassifierCNN  # import your model class


# -------------------------------
# Load Validation Dataset
# -------------------------------
def get_val_loader(data_dir="Spectrograms", batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split into train/val (same as training split)
    _, val_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2,
        random_state=42, stratify=dataset.targets
    )

    val_dataset = Subset(dataset, val_idx)

    # ⚠️ num_workers=0 ensures no multiprocessing
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loader, dataset.classes


# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(model_path="spectogram_classifier.ckpt", data_dir="Spectograms", batch_size=32):
    # Load val data
    val_loader, class_names = get_val_loader(data_dir, batch_size)

    # Load model
    model = SpectogramClassifierCNN.load_from_checkpoint(model_path)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(y.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())

    # Convert to numpy arrays
    all_labels = torch.tensor(all_labels).numpy()
    all_preds = torch.tensor(all_preds).numpy()
    all_probs = torch.tensor(all_probs).numpy()

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    # For ROC-AUC (multi-class with One-vs-Rest)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        roc_auc = None  # e.g. if only one class is present

    # Print results
    print("\n📊 Model Evaluation Results")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC      : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


# -------------------------------
# Main
# -------------------------------
if _name_ == "_main_":
    model_path = "spectogram_classifier.ckpt"  # change if saved elsewhere
    data_dir = "Spectrograms"
    evaluate_model(model_path, data_dir, batch_size=32)