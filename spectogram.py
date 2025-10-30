import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2

# Dataset folders
input_root = "Dataset"
output_root = "Spectrograms"

# Create output root if not exists
os.makedirs(output_root, exist_ok=True)

# Parameters
n_fft = 2048
hop_length = 512
n_mels = 128

def save_spec(aud, sr, save_path):
    """Compute mel-spectrogram, resize, and save as image."""
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=aud, sr=sr,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels,
                                       power=2.0)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Resize to (128,512)
    S_resized = cv2.resize(S_dB, (512, 128))   # width=512, height=128

    # Save as image
    plt.figure(figsize=(5, 2))
    librosa.display.specshow(S_resized, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', fmax=8000)
    plt.axis("off")  # remove axes
    plt.tight_layout(pad=0)

    # Save plot as PNG
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Loop through folders and files
for folder in os.listdir(input_root):
    folder_path = os.path.join(input_root, folder)
    if not os.path.isdir(folder_path):
        continue

    # Create output folder for this class
    output_folder = os.path.join(output_root, folder)
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all .wav files
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            aud, sr = librosa.load(file_path, sr=None)  # keep original sr
            save_name = os.path.splitext(file)[0] + ".png"  # same name, .png extension
            save_path = os.path.join(output_folder, save_name)
            save_spec(aud, sr, save_path)

print("✅ Spectrograms created and saved successfully!")
