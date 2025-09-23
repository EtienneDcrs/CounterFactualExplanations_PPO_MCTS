# -*- coding: utf-8 -*-
import os
import warnings
import logging
import pandas as pd
from typing import Optional, Dict, List, Tuple
import time
from PPO_train import Config, train_ppo_for_counterfactuals, generate_counterfactuals
from utils import get_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def run_ppo_and_metrics():
    """
    Run PPO counterfactual generation on multiple datasets and calculate metrics.
    Saves counterfactuals and original samples, then computes KPIs using utils.get_metrics.
    Measures and reports time taken for counterfactual generation.
    Reports all results at the end.
    
    Returns:
        List of dictionaries containing metrics and generation time for each dataset.
    """
    # Define dataset and model pairs
    datasets = [
        # {
        #     'dataset_name': 'breast_cancer',
        #     'dataset_path': 'data/breast_cancer.csv',
        #     'model_path': 'classification_models/breast_cancer_model.pt',
        #     'constraints': {},
        #     'indices_to_use': Config.INDICES_TO_USE
        # },
        {
            'dataset_name': 'diabetes',
            'dataset_path': 'data/diabetes.csv',
            'model_path': 'classification_models/diabetes_model.pt',
            'constraints': {},
            'indices_to_use': Config.INDICES_TO_USE
        },
        {
            'dataset_name': 'adult',
            'dataset_path': 'data/adult.csv',
            'model_path': 'classification_models/adult_model.pt',
            'constraints': {},
            'indices_to_use': Config.INDICES_TO_USE
        },
        # {
        #     'dataset_name': 'bank',
        #     'dataset_path': 'data/bank.csv',
        #     'model_path': 'classification_models/bank_model.pt',
        #     'constraints': {},
        #     'indices_to_use': Config.INDICES_TO_USE
        # }
    ]
    
    results = []
    
    # Process each dataset
    for dataset in datasets:
        dataset_name = dataset['dataset_name']
        dataset_path = dataset['dataset_path']
        model_path = dataset['model_path']
        constraints = dataset['constraints']
        indices_to_use = dataset['indices_to_use']
        save_path = os.path.join(Config.DATA_DIR, f"generated_counterfactuals_ppo_{dataset_name}.csv")
        
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Validate dataset and model existence
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file {dataset_path} not found. Skipping.")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Dataset file {dataset_path} not found"
            })
            continue
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found. Training a new one.")
        
        # Train or load PPO model
        try:
            ppo_model, env = train_ppo_for_counterfactuals(
                dataset_path=dataset_path,
                model_path=model_path,
                logs_dir=Config.LOGS_DIR,
                save_dir=Config.SAVE_DIR,
                total_timesteps=Config.TOTAL_TIMESTEPS,
                mode=Config.TRAINING_MODE,
                constraints=constraints,
                verbose=1
            )
            logger.info(f"PPO model processed for {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to process PPO model for {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'error': f"Failed to process PPO model: {e}"
            })
            continue
        
        # Generate counterfactuals and measure time
        try:
            start_time = time.time()
            counterfactuals, original_df, counterfactual_df = generate_counterfactuals(
                ppo_model=ppo_model,
                env=env,
                dataset_path=dataset_path,
                save_path=save_path,
                specific_indices=indices_to_use,
                max_steps_per_sample=Config.MAX_STEPS_PER_SAMPLE,
                use_mcts=False,
                mcts_simulations=Config.MCTS_SIMULATIONS,
                verbose=1
            )
            generation_time = time.time() - start_time
            logger.info(f"Counterfactuals generated and saved to {save_path} in {generation_time:.2f} seconds")
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
    run_ppo_and_metrics()