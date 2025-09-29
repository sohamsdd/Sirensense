# Sirensense

### `train.py`
This script handles the full training pipeline, including stratified data splitting, model definition, and checkpoint saving.
It uses PyTorch Lightning to train the `SpectrogramClassifierCNN` and saves the best weights to a `.ckpt` file.
Run this first to generate the necessary `spectrogram_classifier.ckpt` checkpoint file.

### `evaluate.py`
This script loads the trained model from the checkpoint and runs a comprehensive evaluation on the reserved test set.
It calculates and displays key metrics including Accuracy, Precision, Recall, F1-score, and ROC-AUC for all classes.
The script also generates and saves visual plots for the Confusion Matrix and the multi-class ROC Curve.

### `inference_multiple.py`
A standalone utility for loading the trained model checkpoint and performing batch predictions on a list of image files.
It includes the full CNN architecture definition and necessary preprocessing/prediction functions.
Use this to verify model performance on local test data quickly after training.

### `app.py`
The original Streamlit web application for uploading an audio file and performing a single classification.
It internally generates the spectrogram from the audio and uses the trained model to predict the class.
This provides a simple, interactive GUI for testing the model.
