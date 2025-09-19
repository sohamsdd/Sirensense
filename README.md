# Sirensense: Robust Emergency Vehicle Detection Using Synthetic Audio Cues in Urban Noise

We have first implemented the baseline models on MFCC features and then we proceed with more models like CNN using spectrograms.  
There are two branches for this project:  
1. Main
2. Baseline_models  

## Implementation of the baseline models:  

## 1. prepare_dataset.py

**Purpose:**  
Preprocess audio data and extract features for training the model

**Functionality:**  
- Loading raw audio files from the dataset directory  
- Preprocessing the data (trimming, normalization)  
- Extraction of various features such as MFCCs, delata, delta-delta, chroma, spectral contrast  
- Saving processed features  

## 2. split_dataset.py

**Purpose:**  
To split preprocessed data into training, validation, and testing sets.

**Functionality:**  
- Reads the features from prepare_dataset.py
  Divides dataset into:
- Training set: 70% of total data
- Validation set: 15% of total data (for hyperparameter tuning)
- Testing set: 15% of total data (for final evaluation)
- Saves the splits for training, validation, and testing.

## 3. train_classical.py

**Purpose:**  
Train classical ML models.  

**Dataset:**  
Trains on 70% training set 
Tunes hyperparameters using 15% validation set

**Models Implemented:**  
1. k-Nearest Neighbors (kNN):  
   Classifies samples based on nearest neighbors.  

   *Hyperparameters:*  
   n_neighbors (e.g., 5)  
   metric (Euclidean, Manhattan)  

3. Support Vector Machine (SVM):  
   Finds optimal hyperplane to separate classes.  

   *Kernels:*  
   Linear – straight-line boundary  
   Gaussian/RBF – non-linear boundary for complex patterns
   
   *Hyperparameters:*  
   C – regularization strength  
   kernel – linear or rbf(radial basis function)  

3. Random Forest (RF):  
   Ensemble of decision trees voting for class labels  

   *Hyperparameters:*  
   n_estimators – number of trees (200–300)  
   max_depth – maximum depth of each tree  
   max_features – features considered at each split  

## 4. test_classical.py  

**Purpose:**  
Evaluate trained classical ML models on the 15% test set.  

**Functionality:**  
- Loads trained models from models/ folder
- Tests on the testing dataset
- Computes evaluation metrics: accuracy, precision, recall, F1-score, macro/weighted averages
- Generates confusion matrix for performance analysis.
