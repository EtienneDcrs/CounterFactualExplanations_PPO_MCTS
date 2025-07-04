# -*- coding: utf-8 -*-

import os
import warnings
from Classifier_model import load_classifier_with_preprocessing
from CERTIFAI import CERTIFAI 

warnings.filterwarnings("ignore")

# Load dataset
url = 'data/breast_cancer.csv'  # Using breast cancer dataset as specified
url = 'data/diabetes.csv'  # Using breast cancer dataset as specified
url = 'data/adult.csv'  # Using adult dataset as specified

model = 'diabetes_model.pt'
model = 'adult_model.pt'

# Load or train classifier with preprocessing
model_path = os.path.join("classification_models", model)
classifier, label_encoders, scaler = load_classifier_with_preprocessing(url, model_path)

# Generate counterfactual
cert = CERTIFAI.from_csv(url, label_encoders=label_encoders, scaler=scaler)
cert.fit(classifier, generations=15, verbose=True, pytorch=True)