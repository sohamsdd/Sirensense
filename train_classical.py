import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

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

def load_data(data_path):
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

train_path = "dataset/train"
val_path = "dataset/val"

X_train, y_train, classes = load_data(train_path)
X_val, y_val, _ = load_data(val_path)

rf_params = {'n_estimators': [200, 300], 'max_depth': [10, 20, None]}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=3, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_val_pred = rf_best.predict(X_val)
print("Random Forest Validation Accuracy:", rf_best.score(X_val, y_val))
print("RF Classification Report:\n", classification_report(y_val, rf_val_pred, target_names=classes))
joblib.dump(rf_best, "models/rf_model.pkl")

svm_params = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(), svm_params, cv=3, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)
svm_best = svm_grid.best_estimator_
svm_val_pred = svm_best.predict(X_val)
print("SVM Validation Accuracy:", svm_best.score(X_val, y_val))
print("SVM Classification Report:\n", classification_report(y_val, svm_val_pred, target_names=classes))
joblib.dump(svm_best, "models/svm_model.pkl")

knn_params = {'n_neighbors': [5, 10, 15], 'metric': ['euclidean', 'manhattan']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train, y_train)
knn_best = knn_grid.best_estimator_
knn_val_pred = knn_best.predict(X_val)
print("KNN Validation Accuracy:", knn_best.score(X_val, y_val))
print("KNN Classification Report:\n", classification_report(y_val, knn_val_pred, target_names=classes))
joblib.dump(knn_best, "models/knn_model.pkl")
