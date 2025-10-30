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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from typing import List, Union
import time 

# --- 0. Configuration and Constants ---
# NOTE: Place your checkpoint file in the same directory as this script.
CHECKPOINT_PATH = "spectogram_classifier.ckpt" 
SAMPLE_RATE = 16000 # Standard sample rate for speech/sound analysis
CHUNK_DURATION_SECONDS = 5.0 # Process audio in 5-second chunks (must match training data length)
CLASS_NAMES = ["emergency", "urban", "emergency2urban", "urban2emergency"]

# --- 1. Model Architecture (Copied from app.py) ---

class SpectrogramClassifierCNN(pl.LightningModule):
    """
    The CNN model structure used for training.
    """
    def __init__(self, num_classes=4, lr=1e-4):
        super(SpectrogramClassifierCNN, self).__init__()
        # Features and Classifier blocks must match the training script
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

        # Assuming input 128x512
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

    def training_step(self, batch, batch_idx): return None
    def configure_optimizers(self): return None

# --- 2. Utility Functions (Spectrogram/Inference) ---

# Define the transformation pipeline for inference
transform = transforms.Compose([
    transforms.Resize((128, 512)), # MUST match the input size for the CNN
    transforms.ToTensor(),
    # Assuming ImageNet normalization was used in training
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_and_cache_model(checkpoint_path=CHECKPOINT_PATH, num_classes=4):
    """Loads the trained model checkpoint once and caches it."""
    try:
        model = SpectrogramClassifierCNN.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
        model.eval()
        return model
    except FileNotFoundError:
        return None
    except Exception:
        return None

def generate_spectrogram_pil(audio_data, sr):
    """
    Computes mel-spectrogram in memory and returns the PIL image object for inference.
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
    
    return image_pil

def predict_spectrogram(model, image_pil):
    """
    Performs inference on a single PIL image.
    """
    image_tensor = transform(image_pil).unsqueeze(0)
    
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


# --- 3. Real-Time Audio Processor ---

# Global variable to store the latest classification result and status
latest_prediction = {"class": "Waiting for audio...", "probabilities": {}, "status": "Ready"}

class AudioClassifierProcessor(AudioProcessorBase):
    """
    Processes audio chunks from the microphone, converts them to spectrograms,
    and runs inference on the CNN model.
    """
    def __init__(self):
        self.model = load_and_cache_model()
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Calculate the required number of samples for the chunk
        self.samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION_SECONDS)
        print(f"INFO: Target samples per 5s chunk: {self.samples_per_chunk}") # DEBUG PRINT
        
        # Update status if model failed to load
        if self.model is None:
            global latest_prediction
            latest_prediction["status"] = "Model Load Failed"
            latest_prediction["class"] = "ERROR: CHECKPOINT MISSING"
            print("ERROR: Model checkpoint failed to load during processor initialization.")


    # --- Using recv_queued to handle dropped frames ---
    def recv_queued(self, frames: List):
        
        new_samples_list = []
        global latest_prediction

        for frame in frames:
            # Convert frame to numpy array of type float32
            audio_array = frame.to_ndarray(format="flt32")

            # --- DEBUG PRINT: Shows the shape of the incoming audio frame ---
            print(f"DEBUG: Frame shape received: {audio_array.shape}, Frames in queue: {len(frames)}") 
            
            # CRITICAL FIX: Handle stereo audio (2 channels). Our model expects mono (1 channel).
            # The shape should ideally be (N, 1) or just (N,). If it's (N, 2), we slice to mono.
            if audio_array.ndim == 2:
                # Assuming mono is the first channel
                mono_audio = audio_array[:, 0]
            elif audio_array.ndim == 1:
                # Already mono
                mono_audio = audio_array
            else:
                # Unexpected format - skip this frame and log error
                print(f"CRITICAL AUDIO FORMAT ERROR: Unexpected dimensions {audio_array.ndim}")
                continue # Skip to the next frame
            
            new_samples_list.append(mono_audio)


        # Concatenate all new mono samples
        if new_samples_list:
            try:
                new_samples = np.concatenate(new_samples_list)
                self.audio_buffer = np.concatenate([self.audio_buffer, new_samples])
            except Exception as e:
                # Catch errors during NumPy concatenation (should be handled by the logic above, but safer to catch)
                print(f"CRITICAL NUMPY ERROR during concatenation: {e}")
                return frames 
            
            # Explicitly confirm audio is being received
            print(f"Audio frames received (queued mode), new samples added: {len(new_samples)}. Total buffer size: {len(self.audio_buffer)}")
        
        
        # Process the buffer when it has enough data for one chunk
        if len(self.audio_buffer) >= self.samples_per_chunk:
            audio_chunk = self.audio_buffer[:self.samples_per_chunk]
            self.audio_buffer = self.audio_buffer[self.samples_per_chunk:]
            
            if self.model:
                try:
                    latest_prediction["status"] = f"Processing {CHUNK_DURATION_SECONDS}s chunk..."
                    
                    # 1. Generate Spectrogram Image
                    spec_image_pil = generate_spectrogram_pil(audio_chunk, SAMPLE_RATE)
                    
                    # 2. Run Inference
                    predicted_class, probabilities = predict_spectrogram(self.model, spec_image_pil)
                    
                    # 3. Update Global State
                    latest_prediction = {
                        "class": predicted_class,
                        "probabilities": probabilities,
                        "status": f"Predicted ({predicted_class})"
                    }
                except Exception as e:
                    # Catch and report any runtime errors during processing (e.g., model input mismatch)
                    latest_prediction = {
                        "class": "INFERENCE FAILED",
                        "probabilities": {},
                        "status": f"Runtime Error: {str(e)[:50]}..."
                    }
                    print(f"CRITICAL INFERENCE ERROR: {e}")
            
        return frames # Pass the audio frames through unmodified

# --- 4. Streamlit UI ---

def main_live_app():
    st.set_page_config(page_title="Real-Time Audio Classifier", layout="wide")
    st.title("🎤 Real-Time Sound Event Classification")
    st.markdown("Click 'Start' to turn on your microphone. The app analyzes sound in 5-second chunks using **`streamlit-webrtc`**.")

    # Check model loading status before starting
    model_load_status = load_and_cache_model()
    if model_load_status is None:
        st.error("Cannot start. Please ensure `spectrogram_classifier.ckpt` is present and valid.")
        return

    # Configuration to explicitly request the correct sample rate
    media_constraints = {
        "video": False,
        "audio": {
            "sampleRate": SAMPLE_RATE,
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
        },
    }
    
    # CRITICAL FIX: Explicitly configure the WebRTC engine for maximum compatibility
    rtc_configuration = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "sdpSemantics": "unified-plan",
        "offerToReceiveAudio": False, # We only send audio from browser to server
        "offerToReceiveVideo": False,
        "bundlePolicy": "max-bundle",
        "iceTransportPolicy": "all",
        "rtcpMuxPolicy": "require",
        "tcpCandidateOnly": False,
        "iceRestart": False,
    }


    # Container for real-time results display
    results_placeholder = st.empty()
    status_placeholder = st.empty()
    webrtc_state_placeholder = st.empty()


    # --- Start WebRTC Stream ---
    rtc_ctx = webrtc_streamer(
        key="audio_classifier",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioClassifierProcessor,
        media_stream_constraints=media_constraints, # Pass the microphone constraints
        rtc_configuration=rtc_configuration # Pass the low-level WebRTC configuration
    )
    
    # --- Live Result Polling ---
    st.markdown("---")
    st.subheader("Live Classification Output")

    # This loop keeps the Streamlit app refreshing the display with the latest result
    while True:
        # Check and display WebRTC connection status for debugging
        with webrtc_state_placeholder:
            if rtc_ctx.state.playing:
                connection_status = "**Connection:** Connected (Streaming Audio)"
                st.success(connection_status)
            else:
                connection_status = "**Connection:** Closed (Press Start)"
                st.warning(connection_status)


        current_prediction = latest_prediction # Read the global state
        
        with status_placeholder:
            if "status" in current_prediction:
                st.info(f"**Status:** {current_prediction['status']}")
            else:
                st.info("**Status:** Initializing...")


        with results_placeholder.container():
            col_res, col_prob = st.columns([1, 2])
            
            with col_res:
                st.metric(
                    label="Detected Class", 
                    value=current_prediction.get('class', 'N/A'),
                    delta_color="off"
                )

            with col_prob:
                st.markdown("##### Probabilities (%)")
                probs = current_prediction.get('probabilities', {})
                if probs:
                    # Display probabilities as a bar chart
                    df_probs = {
                        'Class': list(probs.keys()),
                        'Probability (%)': list(probs.values())
                    }
                    # Convert to pandas DataFrame for bar_chart, avoiding direct import
                    # Streamlit handles this conversion internally for dicts/lists of dicts if structure is right
                    import pandas as pd
                    df = pd.DataFrame(df_probs)
                    st.bar_chart(df, x='Class', y='Probability (%)', height=250)
                else:
                    st.warning("Awaiting first 5-second chunk...")

        # Control the refresh rate of the Streamlit UI
        time.sleep(1) 


if __name__ == "__main__":
    main_live_app()
