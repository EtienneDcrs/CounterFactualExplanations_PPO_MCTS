# -*- coding: utf-8 -*-

import os
import warnings
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from Classifier_model import Classifier, load_classifier_with_preprocessing, evaluate_model_on_full_dataset
from CERTIFAI import CERTIFAI 

warnings.filterwarnings("ignore")

# Load dataset
url = 'data/diabetes_100.csv'  # Using diabetes dataset as specified
cert = CERTIFAI.from_csv(url)

# Load or train classifier with preprocessing
model_path = os.path.join("classification_models", "diabetes_model.pt")
classifier, label_encoders, scaler = load_classifier_with_preprocessing(url, model_path)

# Generate counterfactual
cert.fit(classifier, generations=15, verbose=True)

# Evaluate robustness and fairness
print("\n(Unnormalised) model's robustness:")
print(cert.check_robustness(), '\n')
print("(Normalised) model's robustness:")
print(cert.check_robustness(normalised=True), '\n')
print("Model's robustness for male subgroup vs. female subgroup (to check fairness of the model):")
print(cert.check_fairness([{'Sex': 'M'}, {'Sex': 'F'}]), '\n')
print("Visualising above results:")
print(cert.check_fairness([{'Sex': 'M'}, {'Sex': 'F'}], visualise_results=True), '\n')
print("Obtain feature importance in the model and visualise:")
print(cert.check_feature_importance(visualise_results=True))