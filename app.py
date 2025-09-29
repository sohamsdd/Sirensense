import streamlit as st
import os
import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image

# --- 1. Model Architecture (Must match the training script exactly) ---

class SpectrogramClassifierCNN(pl.LightningModule):
    """
    The CNN model structure used for training.
    """
    def __init__(self, num_classes=4, lr=1e-4):
        super(SpectrogramClassifierCNN, self).__init__()
        # self.save_hyperparameters() # Required if loading requires hyperparameters, 
                                     # but PyTorch Lightning handles this via checkpoint.

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

        # NOTE: The input size to the linear layer depends on the input image size
        # and the max pooling steps. Assuming input 128x512:
        # 128 / (2^5) = 4 (Height)
        # 512 / (2^5) = 16 (Width)
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(512 * 4 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # Required placeholder methods for PyTorch Lightning, even if not used for inference
    def training_step(self, batch, batch_idx): return None
    def configure_optimizers(self): return None

# --- 2. Spectrogram Generation Function (Adapted from Spectogram.py) ---

@st.cache_data
def generate_spectrogram_from_audio(audio_data, sr):
    """
    Computes mel-spectrogram in memory and returns the image buffer and PIL image.
    This uses the exact logic (n_mels, resizing) from Spectogram.py.
    """
    n_fft = 2048
    hop_length = 512
    n_mels = 128

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=audio_data, sr=sr,
                                       n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_mels, power=2.0)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Resize to (128, 512) array shape using cv2
    S_resized = cv2.resize(S_dB, (512, 128))  # width=512, height=128

    # Generate the plot into an in-memory buffer
    buf = io.BytesIO()
    plt.figure(figsize=(5, 2))
    librosa.display.specshow(S_resized, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', fmax=8000)
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    # Save plot as PNG to the buffer
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Get the image back as a PIL object for model preprocessing
    buf.seek(0)
    image_pil = Image.open(buf).convert("RGB")
    
    return buf.getvalue(), image_pil # Returns (PNG bytes, PIL image)

# --- 3. Inference Function (Adapted from inference_multiple.py) ---

# Define the transformation pipeline for inference
transform = transforms.Compose([
    transforms.Resize((128, 512)), # MUST match the input size for the CNN
    transforms.ToTensor(),
    # NOTE: Normalization is CRUCIAL. Assuming ImageNet normalization was used in training.
    # If a custom normalization was used, please replace these values.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ["emergency", "emergency2urban", "urban", "urban2emergency"]

@st.cache_resource
def load_and_cache_model(checkpoint_path="spectogram_classifier.ckpt", num_classes=4):
    """Loads the trained model checkpoint once and caches it."""
    try:
        model = SpectrogramClassifierCNN.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Error: Model checkpoint file '{checkpoint_path}' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_spectrogram(model, image_pil):
    """
    Performs inference on a single PIL image.
    """
    image_tensor = transform(image_pil).unsqueeze(0)  # Transform and add batch dimension
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        
        # Convert to percentages
        probs_percent = [float(p) * 100 for p in probs]
        
        # Pair class with probability
        results = {cls: round(p, 2) for cls, p in zip(CLASS_NAMES, probs_percent)}

        predicted_idx = probs.argmax()
        predicted_class = CLASS_NAMES[predicted_idx]
        
        return predicted_class, results

# --- 4. Streamlit Application ---

def main():
    st.set_page_config(page_title="Spectrogram Audio Classifier", layout="wide")
    st.title("🔊 Spectrogram-Based Audio Classifier")
    st.markdown("Upload an audio file (.wav, .mp3) to generate its spectrogram and classify the sound.")
    
    # Load the model
    model = load_and_cache_model()
    if model is None:
        st.warning("Model could not be loaded. Please ensure 'spectogram_classifier.ckpt' is present.")
        return

    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file (e.g., .wav or .mp3)", 
        type=['wav', 'mp3', 'flac']
    )

    if uploaded_file is not None:
        try:
            # Display file info
            st.subheader(f"Processing: {uploaded_file.name}")
            
            # Load audio using librosa from the uploaded file buffer
            # We use a temporary file path from the uploaded object's buffer
            audio_data, sr = librosa.load(io.BytesIO(uploaded_file.getvalue()), sr=None)

            # Generate Spectrogram
            with st.spinner("Generating Spectrogram..."):
                png_bytes, spec_image_pil = generate_spectrogram_from_audio(audio_data, sr)
            
            # Display Spectrogram and Inference
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Generated Spectrogram")
                st.image(png_bytes, caption="Input Spectrogram Image", use_column_width=True)

            with col2:
                st.subheader("Classification Result")
                with st.spinner("Running Inference..."):
                    predicted_class, probabilities = predict_spectrogram(model, spec_image_pil)
                
                # Display Prediction
                st.success(f"**Predicted Class:** {predicted_class}")
                st.markdown(f"The model is highly confident that this audio belongs to the **{predicted_class.upper()}** class.")
                
                # Display Probabilities
                st.markdown("---")
                st.markdown("#### Class Probabilities (%)")
                
                # Sort probabilities for display
                sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

                for cls, prob in sorted_probs.items():
                    color = "green" if cls == predicted_class else "blue"
                    st.markdown(f"<span style='color:{color}; font-weight:bold;'>{cls.ljust(20)}:</span> {prob:.2f}%", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.info("Please ensure your audio file is valid and the model checkpoint matches the architecture.")


if __name__ == "__main__":
    main()
