# -*- coding: utf-8 -*-
import os
import warnings
from Classifier_model import load_classifier_with_preprocessing
from CERTIFAI import CERTIFAI
from utils import get_metrics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def run_certifai_and_metrics():
    """
    Run CERTIFAI counterfactual generation on multiple datasets and calculate metrics.
    Saves counterfactuals and original samples, then computes KPIs using utils.get_metrics.
    """
    # Define dataset and model pairs
    datasets = [
        {'url': 'data/breast_cancer.csv', 'model': 'breast_cancer_model.pt', 'constraints': {}},
        {'url': 'data/diabetes.csv', 'model': 'diabetes_model.pt', 'constraints': {}},
        {'url': 'data/adult.csv', 'model': 'adult_model.pt', 'constraints': {'sex': 'fixed'}}
    ]
    datasets = [
        {'url': 'data/bank.csv', 'model': 'bank_model.pt', 'constraints': {}}
    ]
    
    # Process each dataset
    for dataset in datasets:
        url = dataset['url']
        model = dataset['model']
        constraints = dataset['constraints']
        dataset_name = os.path.splitext(os.path.basename(url))[0]
        output_csv = f"certifai_{dataset_name}.csv"
        
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Validate dataset existence
        if not os.path.exists(url):
            logger.error(f"Dataset file {url} not found. Skipping.")
            continue
        
        # Load or train classifier with preprocessing
        model_path = os.path.join("classification_models", model)
        try:
            classifier, label_encoders, scaler = load_classifier_with_preprocessing(url, model_path)
            logger.info(f"Loaded classifier for {dataset_name} from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load classifier for {dataset_name}: {e}")
            continue
        
        # Generate counterfactuals
        try:
            cert = CERTIFAI.from_csv(url, label_encoders=label_encoders, scaler=scaler)
            cert.fit(classifier, generations=15, verbose=True, pytorch=True)
            cert.save_counterfactuals_to_csv(output_csv)
            logger.info(f"Counterfactuals generated and saved to {output_csv}")
        except Exception as e:
            logger.error(f"Failed to generate counterfactuals for {dataset_name}: {e}")
            continue
        
        # Calculate metrics
        original_csv = f"{os.path.splitext(output_csv)[0]}_original.csv"
        try:
            coverage, distance, implausibility, sparsity, actionability, diversity = get_metrics(
                original_csv_path=original_csv,
                counterfactual_csv_path=output_csv,
                constraints=constraints
            )
            logger.info(f"\nMetrics for {dataset_name}:")
            logger.info(f"Coverage: {coverage:.2%}")
            logger.info(f"Mean distance: {distance:.2f}")
            logger.info(f"Mean implausibility: {implausibility:.2f}")
            logger.info(f"Mean sparsity: {sparsity:.2f}")
            logger.info(f"Mean actionability: {actionability:.2f}")
            logger.info(f"Diversity: {diversity:.2f}")
        except Exception as e:
            logger.error(f"Failed to calculate metrics for {dataset_name}: {e}")

if __name__ == "__main__":
    run_certifai_and_metrics()