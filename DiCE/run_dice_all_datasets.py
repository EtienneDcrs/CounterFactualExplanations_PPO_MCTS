# -*- coding: utf-8 -*-
import os
import warnings
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
import pickle
from sklearn.model_selection import train_test_split
import dice_ml
import time
from utils import get_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Classifier class (unchanged)
class Classifier(pl.LightningModule):
    def __init__(self, in_feats=8, h_size=128, out=2, n_layers=4,
                 activation_function=nn.ReLU, lr=1e-3, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        mlp_sequence = OrderedDict([('mlp1', nn.Linear(in_features=in_feats,
                                                      out_features=h_size)),
                                   ('activ1', activation_function()),
                                   ('dropout1', nn.Dropout(dropout))])
        for i in range(1, n_layers):
            new_keys = ['mlp' + str(i + 1), 'activ' + str(i + 1), 'dropout' + str(i + 1)]
            mlp_sequence[new_keys[0]] = nn.Linear(in_features=h_size, out_features=h_size)
            mlp_sequence[new_keys[1]] = activation_function()
            mlp_sequence[new_keys[2]] = nn.Dropout(dropout)
        mlp_sequence['out_projection'] = nn.Linear(in_features=h_size, out_features=out)
        self.net = nn.Sequential(mlp_sequence)
        self.learning_rate = lr
        self.weight_decay = weight_decay

    def forward(self, x, apply_softmax=False):
        y = self.net(x)
        if apply_softmax:
            return nn.functional.softmax(y, -1)
        return y

# PyTorchModelWrapper class (unchanged)
class PyTorchModelWrapper:
    def __init__(self, model_path, encoder, scaler, feature_names, continuous_features):
        input_dim = len(feature_names)
        self.model = Classifier(in_feats=input_dim, h_size=128, out=2, n_layers=4,
                                activation_function=nn.ReLU, dropout=0.3, lr=0.001, weight_decay=1e-4)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.encoder = encoder
        self.scaler = scaler
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features = [f for f in feature_names if f not in continuous_features]

    def __call__(self, X):
        if isinstance(X, pd.DataFrame):
            return torch.tensor(self.predict_proba(X), dtype=torch.float32)
        return self.model(X, apply_softmax=True)

    def predict_proba(self, X):
        X_processed = X.copy()
        for col in self.categorical_features:
            if col in X_processed.columns and col in self.encoder:
                X_processed[col] = self.encoder[col].transform(X_processed[col])
        if self.continuous_features:
            X_processed[self.continuous_features] = self.scaler.transform(X_processed[self.continuous_features])
        X_tensor = torch.tensor(X_processed[self.feature_names].values, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_tensor, apply_softmax=True).numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Utility functions (unchanged)
def load_custom_dataset(file_path, target_column):
    """
    Load dataset from a CSV file.
    """
    dataset = pd.read_csv(file_path)
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    return dataset

def load_preprocessors(encoder_path, scaler_path):
    """
    Load encoders and scalers from .pkl files.
    """
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return encoder, scaler

def preprocess_dataset_robust(dataset, encoder, scaler, feature_names, continuous_features, target_column):
    """
    Robust preprocessing for DiCE compatibility.
    """
    dataset_processed = dataset.copy()
    categorical_features = [f for f in feature_names if f not in continuous_features]
    
    logger.debug(f"Preprocessing dataset. Shape: {dataset_processed.shape}, Columns: {dataset_processed.columns.tolist()}")
    logger.debug(f"Feature names: {feature_names}, Continuous: {continuous_features}, Categorical: {categorical_features}")
    
    # Encode categorical features
    for col in categorical_features:
        if col in dataset_processed.columns and col in encoder:
            try:
                dataset_processed[col] = encoder[col].transform(dataset_processed[col])
                logger.debug(f"Encoded categorical feature: {col}")
            except Exception as e:
                logger.warning(f"Failed to encode {col}: {e}")
    
    # Scale continuous features
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = scaler.feature_names_in_
        available_features = [f for f in expected_features if f in dataset_processed.columns and f != target_column]
        if available_features:
            try:
                dataset_processed[available_features] = scaler.transform(dataset_processed[available_features])
                logger.debug(f"Scaled features: {available_features}")
            except Exception as e:
                logger.warning(f"Failed to scale with expected features: {e}")
        else:
            logger.warning("No available features match scaler expectations")
    else:
        available_continuous = [f for f in continuous_features if f in dataset_processed.columns]
        if available_continuous:
            try:
                dataset_processed[available_continuous] = scaler.transform(dataset_processed[available_continuous])
                logger.debug(f"Scaled continuous features: {available_continuous}")
            except Exception as e:
                logger.warning(f"Failed to scale continuous features: {e}")
    
    return dataset_processed

def save_counterfactuals_to_csv(counterfactuals, original_df, counterfactual_df, save_path, encoder, scaler, continuous_features):
    """
    Save counterfactuals and original samples to CSV files in a format compatible with utils.get_metrics.
    Decodes categorical features and unscales continuous features, handling out-of-range values.
    
    Args:
        counterfactuals: List of dictionaries containing sample_id, counterfactual_features, success.
        original_df: DataFrame of original samples (preprocessed).
        counterfactual_df: DataFrame of counterfactual samples (preprocessed).
        save_path: Path to save the counterfactual CSV file (original CSV will append '_original').
        encoder: Dictionary of LabelEncoder objects for categorical features.
        scaler: Scaler object for continuous features.
        continuous_features: List of continuous feature names.
    """
    counterfactual_rows = []
    original_rows = []
    feature_columns = original_df.columns.tolist()
    categorical_features = [col for col in feature_columns if col not in continuous_features]
    
    # Prepare inverse-transformed DataFrames
    original_df_decoded = original_df.copy()
    counterfactual_df_decoded = counterfactual_df.copy()
    
    # Inverse-transform categorical features with clamping for out-of-range values
    for col in categorical_features:
        if col in encoder:
            try:
                # Get valid encoded values from encoder
                valid_labels = np.arange(len(encoder[col].classes_))
                # Clamp counterfactual values to valid range
                counterfactual_df_decoded[col] = np.clip(counterfactual_df_decoded[col].astype(int), 
                                                         valid_labels.min(), valid_labels.max())
                original_df_decoded[col] = np.clip(original_df_decoded[col].astype(int), 
                                                  valid_labels.min(), valid_labels.max())
                # Decode
                original_df_decoded[col] = encoder[col].inverse_transform(original_df_decoded[col])
                counterfactual_df_decoded[col] = encoder[col].inverse_transform(counterfactual_df_decoded[col])
            except Exception as e:
                logger.warning(f"Failed to decode categorical feature {col}: {e}")
                original_df_decoded[col] = original_df[col]
                counterfactual_df_decoded[col] = counterfactual_df[col]
    
    # Inverse-scale continuous features
    if continuous_features and hasattr(scaler, 'inverse_transform'):
        try:
            available_continuous = [col for col in continuous_features if col in original_df.columns]
            if available_continuous:
                if hasattr(scaler, 'feature_names_in_') and len(scaler.feature_names_in_) > len(continuous_features):
                    # Scaler was fitted on all features; select continuous features
                    scaler_indices = [list(scaler.feature_names_in_).index(col) for col in available_continuous]
                    original_scaled = scaler.inverse_transform(original_df[scaler.feature_names_in_])
                    counterfactual_scaled = scaler.inverse_transform(counterfactual_df[scaler.feature_names_in_])
                    for idx, col in enumerate(available_continuous):
                        original_df_decoded[col] = original_scaled[:, scaler_indices[idx]]
                        counterfactual_df_decoded[col] = counterfactual_scaled[:, scaler_indices[idx]]
                else:
                    # Scaler was fitted only on continuous features
                    original_df_decoded[available_continuous] = scaler.inverse_transform(original_df[available_continuous])
                    counterfactual_df_decoded[available_continuous] = scaler.inverse_transform(counterfactual_df[available_continuous])
        except Exception as e:
            logger.warning(f"Failed to inverse-scale continuous features: {e}")
            # Fall back to original values for continuous features
            original_df_decoded[available_continuous] = original_df[available_continuous]
            counterfactual_df_decoded[available_continuous] = counterfactual_df[available_continuous]
    
    for i, cf in enumerate(counterfactuals):
        sample_id = cf['sample_id']
        # Save original sample
        original_row = {col: original_df_decoded.iloc[i][col] for col in feature_columns}
        original_row['sample_id'] = sample_id
        original_rows.append(original_row)
        
        # Save counterfactual or original features if no counterfactual
        if cf['success'] and cf['counterfactual_features'] is not None:
            row = {col: counterfactual_df_decoded.iloc[i][col] for col in feature_columns}
            row['sample_id'] = sample_id
            row['counterfactual_found'] = 1
        else:
            row = {col: original_df_decoded.iloc[i][col] for col in feature_columns}
            row['sample_id'] = sample_id
            row['counterfactual_found'] = 0
        counterfactual_rows.append(row)
    
    # Create DataFrames
    counterfactual_df_out = pd.DataFrame(counterfactual_rows)
    original_df_out = pd.DataFrame(original_rows)
    
    # Save to CSVs
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    counterfactual_df_out.to_csv(save_path, index=False)
    original_filename = f"{os.path.splitext(save_path)[0]}_original.csv"
    original_df_out.to_csv(original_filename, index=False)
    
    logger.info(f"Counterfactuals saved to {save_path}")
    logger.info(f"Original samples saved to {original_filename}")

def run_dice_and_metrics():
    """
    Run DiCE counterfactual generation on multiple datasets and calculate metrics.
    Saves counterfactuals and original samples, then computes KPIs using utils.get_metrics.
    Measures and reports time taken for counterfactual generation.
    Reports all results at the end.
    
    Returns:
        List of dictionaries containing metrics and generation time for each dataset.
    """
    datasets = [
        # {
        #     'dataset_name': 'bank',
        #     'csv_file_path': 'data/bank.csv',
        #     'target_column': 'y',
        #     'continuous_features': ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'],
        #     'categorical_features': ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'],
        #     'model_path': 'classification_models/bank_model.pt',
        #     'encoder_path': 'classification_models/bank_model_encoders.pkl',
        #     'scaler_path': 'classification_models/bank_model_scaler.pkl',
        #     'constraints': {}
        # },
        # {
        #     'dataset_name': 'diabetes',
        #     'csv_file_path': 'data/diabetes.csv',
        #     'target_column': 'Outcome',
        #     'continuous_features': ['Pregnancies', 'Glucose', 'BloodPressure',
        #                               'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        #     'categorical_features': [],
        #     'model_path': 'classification_models/diabetes_model.pt',
        #     'encoder_path': 'classification_models/diabetes_model_encoders.pkl',
        #     'scaler_path': 'classification_models/diabetes_model_scaler.pkl',
        #     'constraints': {}
        # },
        # {
        #     'dataset_name': 'adult',
        #     'csv_file_path': 'data/adult.csv',
        #     'target_column': 'income',
        #     'continuous_features': ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
        #     'categorical_features': ['workclass', 'education', 'marital-status', 'occupation', 'relationship','race','sex','native-country'],
        #     'model_path': 'classification_models/adult_model.pt',
        #     'encoder_path': 'classification_models/adult_model_encoders.pkl',
        #     'scaler_path': 'classification_models/adult_model_scaler.pkl',
        #     'constraints': {}                            
        # },
        {
            'dataset_name': 'breast_cancer',
            'csv_file_path': 'data/breast_cancer.csv',
            'target_column': 'irradiat',
            'continuous_features': [],
            'categorical_features': ['class', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad'],
            'model_path': 'classification_models/breast_cancer_model.pt',
            'encoder_path': 'classification_models/breast_cancer_model_encoders.pkl',
            'scaler_path': 'classification_models/breast_cancer_model_scaler.pkl',
            'constraints': {}
        }
    ]
    
    results = []
    
    for dataset in datasets:
        dataset_name = dataset['dataset_name']
        csv_file_path = dataset['csv_file_path']
        target_column = dataset['target_column']
        continuous_features = dataset['continuous_features']
        categorical_features = dataset['categorical_features']
        model_path = dataset['model_path']
        encoder_path = dataset['encoder_path']
        scaler_path = dataset['scaler_path']
        constraints = dataset['constraints']
        save_path = f"data/generated_counterfactuals_dice_{dataset_name}.csv"
        
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Validate file existence
        if not os.path.exists(csv_file_path):
            logger.error(f"Dataset file {csv_file_path} not found. Skipping.")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Dataset file {csv_file_path} not found"
            })
            continue
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found. Skipping.")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Model file {model_path} not found"
            })
            continue
        if not os.path.exists(encoder_path) or not os.path.exists(scaler_path):
            logger.error(f"Encoder ({encoder_path}) or scaler ({scaler_path}) not found. Skipping.")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Encoder or scaler file not found"
            })
            continue
        
        # Load dataset
        try:
            dataset_df = load_custom_dataset(csv_file_path, target_column)
            feature_names = [col for col in dataset_df.columns if col != target_column]
            logger.info(f"Loaded dataset {dataset_name} with {len(dataset_df)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Failed to load dataset: {e}"
            })
            continue
        
        # Load preprocessors
        try:
            encoder, scaler = load_preprocessors(encoder_path, scaler_path)
            logger.info(f"Loaded preprocessors for {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load preprocessors for {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Failed to load preprocessors: {e}"
            })
            continue
        
        # Preprocess dataset
        try:
            dataset_processed = preprocess_dataset_robust(dataset_df, encoder, scaler, feature_names, continuous_features, target_column)
            logger.info(f"Preprocessed dataset {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to preprocess dataset {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Failed to preprocess dataset: {e}"
            })
            continue
        
        # Split dataset
        try:
            train_dataset, test_dataset, _, _ = train_test_split(
                dataset_processed,
                dataset_processed[target_column],
                test_size=0.2,
                random_state=0,
                stratify=dataset_processed[target_column]
            )
            logger.info(f"Split dataset {dataset_name}: {len(train_dataset)} train, {len(test_dataset)} test")
        except Exception as e:
            logger.error(f"Failed to split dataset {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Failed to split dataset: {e}"
            })
            continue
        
        # Prepare DiCE
        try:
            d = dice_ml.Data(dataframe=train_dataset,
                            continuous_features=continuous_features,
                            outcome_name=target_column)
            model_wrapper = PyTorchModelWrapper(model_path, encoder, scaler, feature_names, continuous_features)
            m = dice_ml.Model(model=model_wrapper, backend='PYT')
            exp = dice_ml.Dice(d, m)
            logger.info(f"Initialized DiCE for {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to initialize DiCE for {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Failed to initialize DiCE: {e}"
            })
            continue
        
        # Generate counterfactuals for first 100 instances and measure time
        test_instances = test_dataset.drop(columns=target_column)
        if len(test_instances) < 100:
            test_instances = test_instances.sample(n=100, replace=True, random_state=0)
        else:
            test_instances = test_instances.iloc[:100]
        
        counterfactuals = []
        original_samples = []
        counterfactual_samples = []
        
        try:
            start_time = time.time()
            for idx, (original_idx, row) in enumerate(test_instances.iterrows()):
                query_instance = row.to_frame().T
                try:
                    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=1, desired_class="opposite")
                    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                    success = not cf_df.empty
                    cf_instance = cf_df.drop(columns=target_column).iloc[0].to_numpy() if success else None
                    counterfactuals.append({
                        'sample_id': idx,
                        'counterfactual_features': cf_instance,
                        'success': success
                    })
                    original_samples.append(query_instance.iloc[0].to_numpy())
                    counterfactual_samples.append(cf_instance if success else query_instance.iloc[0].to_numpy())
                    logger.debug(f"Generated counterfactual for instance {idx+1} in {dataset_name}: {'Success' if success else 'Failed'}")
                except Exception as e:
                    logger.warning(f"Failed to generate counterfactual for instance {idx+1} in {dataset_name}: {e}")
                    counterfactuals.append({
                        'sample_id': idx,
                        'counterfactual_features': None,
                        'success': False
                    })
                    original_samples.append(query_instance.iloc[0].to_numpy())
                    counterfactual_samples.append(query_instance.iloc[0].to_numpy())
            generation_time = time.time() - start_time
            logger.info(f"Counterfactuals generated and saved to {save_path} in {generation_time:.2f} seconds")
            
            original_df = pd.DataFrame(original_samples, columns=feature_names)
            counterfactual_df = pd.DataFrame(counterfactual_samples, columns=feature_names)
            save_counterfactuals_to_csv(counterfactuals, original_df, counterfactual_df, save_path, encoder, scaler, continuous_features)
        except Exception as e:
            logger.error(f"Failed to generate counterfactuals for {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Failed to generate counterfactuals: {e}"
            })
            continue
        
        # Calculate metrics
        original_csv = f"{os.path.splitext(save_path)[0]}_original.csv"
        try:
            coverage, distance, implausibility, sparsity, actionability, diversity = get_metrics(
                original_csv_path=original_csv,
                counterfactual_csv_path=save_path,
                constraints=constraints
            )
            results.append({
                'dataset_name': dataset_name,
                'coverage': coverage,
                'distance': distance,
                'implausibility': implausibility,
                'sparsity': sparsity,
                'actionability': actionability,
                'diversity': diversity,
                'generation_time_seconds': generation_time
            })
            logger.info(f"Metrics calculated for {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to calculate metrics for {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Failed to calculate metrics: {e}"
            })
    
    # Log all results at the end
    logger.info("\n=== Final Results ===")
    for result in results:
        logger.info(f"\nDataset: {result['dataset_name']}")
        if 'error' in result:
            logger.info(f"Error: {result['error']}")
        else:
            logger.info(f"Coverage: {result['coverage']:.2%}")
            logger.info(f"Mean distance: {result['distance']:.2f}")
            logger.info(f"Mean implausibility: {result['implausibility']:.2f}")
            logger.info(f"Mean sparsity: {result['sparsity']:.2f}")
            logger.info(f"Mean actionability: {result['actionability']:.2f}")
            logger.info(f"Diversity: {result['diversity']:.2f}")
            logger.info(f"Generation time: {result['generation_time_seconds']:.2f} seconds")

    return results

if __name__ == "__main__":
    run_dice_and_metrics()