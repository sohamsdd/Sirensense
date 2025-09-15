import os
import numpy as np
import librosa
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    features = np.concatenate((mfcc, delta, delta2, chroma, spec_contrast), axis=0)
    
    return np.mean(features.T, axis=0)

def load_test_data(data_path):
    X, y = [], []
    classes = sorted(os.listdir(data_path))
    for label, cls in enumerate(classes):
        cls_path = os.path.join(data_path, cls)
        for file in os.listdir(cls_path):
            if file.endswith('.wav'):
                file_path = os.path.join(cls_path, file)
                feats = extract_features(file_path)
                X.append(feats)
                y.append(label)
    return np.array(X), np.array(y), classes

test_path = "dataset/test"
results_path = "results"
os.makedirs(results_path, exist_ok=True)

X_test, y_test, classes = load_test_data(test_path)

models = {
    "Random Forest": joblib.load("models/rf_model.pkl"),
    "SVM": joblib.load("models/svm_model.pkl"),
    "KNN": joblib.load("models/knn_model.pkl")
}

for name, model in models.items():
    print(f"\n--- {name} Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(f"{name} Test Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    file_name = os.path.join(results_path, f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.savefig(file_name)
    plt.close()
    print(f"Confusion matrix saved to {file_name}")
