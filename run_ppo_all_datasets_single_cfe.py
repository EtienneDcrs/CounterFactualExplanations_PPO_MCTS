# -*- coding: utf-8 -*-
import os
import warnings
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import time
from PPO_train import Config, train_ppo_for_counterfactuals, generate_multiple_counterfactuals_for_sample, get_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def run_ppo_and_metrics():
    """
    Run PPO counterfactual generation on multiple datasets and calculate metrics.
    For each dataset, generates multiple counterfactuals for several samples and averages the metrics.
    Saves counterfactuals and original samples for each sample, then computes KPIs using PPO_train.get_metrics.
    Measures and reports average time taken for counterfactual generation.
    Reports averaged results at the end.
    
    Returns:
        List of dictionaries containing averaged metrics and generation time for each dataset.
    """
    # Define dataset and model pairs
    datasets = [
        # {
        #     'dataset_name': 'breast_cancer',
        #     'dataset_path': 'data/breast_cancer.csv',
        #     'model_path': 'classification_models/breast_cancer_model.pt',
        #     'constraints': {},
        # },
        # {
        #     'dataset_name': 'diabetes',
        #     'dataset_path': 'data/diabetes.csv',
        #     'model_path': 'classification_models/diabetes_model.pt',
        #     'constraints': {},
        # },
        # {
        #     'dataset_name': 'adult',
        #     'dataset_path': 'data/adult.csv',
        #     'model_path': 'classification_models/adult_model.pt',
        #     'constraints': {},
        # },
        {
            'dataset_name': 'bank',
            'dataset_path': 'data/bank.csv',
            'model_path': 'classification_models/bank_model.pt',
            'constraints': {},
        }
    ]
    
    # Parameters for multiple counterfactuals and samples
    num_samples = 10
    num_counterfactuals = 20
    
    results = []
    
    # Process each dataset
    for dataset in datasets:
        dataset_name = dataset['dataset_name']
        dataset_path = dataset['dataset_path']
        model_path = dataset['model_path']
        constraints = dataset['constraints']
        
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
        
        # Train or load PPO model (shared for all samples in the dataset)
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
        
        # Load the dataset to get its length and feature columns
        full_dataset = pd.read_csv(dataset_path)
        dataset_length = len(full_dataset)
        feature_columns = full_dataset.columns[:-1].tolist()  # Exclude target column
        
        # Select random sample indices (without replacement, up to num_samples)
        sample_indices = np.random.choice(dataset_length, min(num_samples, dataset_length), replace=False)
        
        metrics_per_sample = []
        generation_times = []
        
        for sample_index in sample_indices:
            save_path = os.path.join(Config.DATA_DIR, f"generated_counterfactuals_ppo_{dataset_name}_sample_{sample_index}.csv")
            
            # Generate multiple counterfactuals and measure time
            try:
                start_time = time.time()
                counterfactuals, original_df, counterfactual_df = generate_multiple_counterfactuals_for_sample(
                    ppo_model=ppo_model,
                    env=env,
                    dataset_path=dataset_path,
                    sample_index=sample_index,
                    num_counterfactuals=num_counterfactuals,
                    save_path=save_path,
                    max_steps_per_sample=Config.MAX_STEPS_PER_SAMPLE,
                    use_mcts=False,
                    mcts_simulations=Config.MCTS_SIMULATIONS,
                    verbose=1
                )
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                logger.info(f"{num_counterfactuals} counterfactuals for sample {sample_index} generated and saved to {save_path} in {generation_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to generate counterfactuals for {dataset_name} sample {sample_index}: {e}")
                continue
            
            # Calculate metrics for this sample's counterfactuals using PPO_train.get_metrics
            try:
                coverage, distance, implausibility, sparsity, actionability = get_metrics(
                    original_df=original_df,
                    counterfactual_df=counterfactual_df,
                    counterfactuals=counterfactuals,
                    constraints=constraints,
                    feature_columns=feature_columns,
                    original_data=full_dataset,
                    verbose=1
                )
                from PPO_train import get_diversity
                diversity = get_diversity(counterfactual_df)
                
                metrics_per_sample.append({
                    'coverage': coverage,
                    'distance': distance,
                    'implausibility': implausibility,
                    'sparsity': sparsity,
                    'actionability': actionability,
                    'diversity': diversity
                })
                logger.info(f"Metrics calculated for {dataset_name} sample {sample_index}")
            except Exception as e:
                logger.error(f"Failed to calculate metrics for {dataset_name} sample {sample_index}: {e}")
                continue
        
        # Compute average metrics if any samples were processed
        if metrics_per_sample:
            # Filter out samples with invalid (infinite) distances
            valid_metrics = [m for m in metrics_per_sample if np.isfinite(m['distance'])]
            if valid_metrics:
                avg_metrics = {
                    'dataset_name': dataset_name,
                    'num_samples': len(valid_metrics),
                    'num_counterfactuals_per_sample': num_counterfactuals,
                    'coverage': np.mean([m['coverage'] for m in valid_metrics]),
                    'distance': np.mean([m['distance'] for m in valid_metrics]),
                    'implausibility': np.mean([m['implausibility'] for m in valid_metrics]),
                    'sparsity': np.mean([m['sparsity'] for m in valid_metrics]),
                    'actionability': np.mean([m['actionability'] for m in valid_metrics]),
                    'diversity': np.mean([m['diversity'] for m in valid_metrics]),
                    'average_generation_time_seconds': np.mean(generation_times)
                }
                results.append(avg_metrics)
            else:
                results.append({
                    'dataset_name': dataset_name,
                    'error': "No samples had valid metrics"
                })
        else:
            results.append({
                'dataset_name': dataset_name,
                'error': "No samples processed successfully"
            })
    
    # Log all results at the end
    logger.info("\n=== Final Results ===")
    for result in results:
        logger.info(f"\nDataset: {result['dataset_name']}")
        if 'error' in result:
            logger.info(f"Error: {result['error']}")
        else:
            logger.info(f"Number of Samples: {result['num_samples']}")
            logger.info(f"Counterfactuals per Sample: {result['num_counterfactuals_per_sample']}")
            logger.info(f"Average Coverage: {result['coverage']:.2%}")
            logger.info(f"Average Mean distance: {result['distance']:.2f}")
            logger.info(f"Average Mean implausibility: {result['implausibility']:.2f}")
            logger.info(f"Average Mean sparsity: {result['sparsity']:.2f}")
            logger.info(f"Average Mean actionability: {result['actionability']:.2f}")
            logger.info(f"Average Diversity: {result['diversity']:.2f}")
            logger.info(f"Average Generation time per sample: {result['average_generation_time_seconds']:.2f} seconds")

    return results

if __name__ == "__main__":
    run_ppo_and_metrics()