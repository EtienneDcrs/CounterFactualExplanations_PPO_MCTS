import os
import time
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from PPO_env import CERTIFAIEnv, CERTIFAIMonitorCallback
from ModelExCERTIFAI_PPO import Classifier

def train_ppo_for_counterfactuals(dataset_path, model_path=None, logs_dir='ppo_logs', 
                                  save_dir='ppo_models', total_timesteps=100000, 
                                  continue_training=True):
    """
    Train a PPO agent to generate counterfactuals for a given classifier model.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset CSV file
    model_path : str, optional
        Path to the pre-trained classifier model
    logs_dir : str
        Directory for storing tensorboard logs
    save_dir : str
        Directory for saving PPO model checkpoints
    total_timesteps : int
        Total number of training timesteps
    continue_training : bool
        Whether to continue training an existing PPO model if available
        
    Returns:
    --------
    model : PPO
        Trained PPO model
    env : CERTIFAIEnv
        Environment used for training
    """
    print(f"Starting counterfactual generation with PPO for dataset: {dataset_path}")
    
    # Create directories if they don't exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine the output size based on the dataset
    if 'adult' in dataset_path:
        out = 2
    elif 'drug' in dataset_path:
        out = 5
    elif 'iris' in dataset_path:
        out = 3
    elif 'diabetes' in dataset_path:
        out = 2
    else:
        # Default case - try to determine from the dataset
        dataset = pd.read_csv(dataset_path)
        out = len(dataset.iloc[:, -1].unique())
    
    # Load or train the classifier model
    if model_path is None:
        model_path = f"{os.path.splitext(os.path.basename(dataset_path))[0]}_model.pt"
    
    # Check if the classifier model exists, otherwise exit
    if os.path.exists(model_path):
        print(f"Loading classifier model from {model_path}")
        # Load dataset to determine input features
        dataset = pd.read_csv(dataset_path)
        in_feats = len(dataset.columns) - 1  # Exclude target variable
        
        # Load the classifier model
        classifier = Classifier(in_feats=in_feats, out=out)
        classifier.load_state_dict(torch.load(model_path))
        classifier.eval()  # Set to evaluation mode
    else:
        print(f"Classifier model not found at {model_path}. Please train the classifier first.")
        return None, None
    
    # Create the CERTIFAI environment
    env = CERTIFAIEnv(dataset_path=dataset_path, model=classifier)
    eval_env = CERTIFAIEnv(dataset_path=dataset_path, model=classifier)
    
    # Define PPO model path
    ppo_model_path = os.path.join(save_dir, f"ppo_certifai_final_{os.path.basename(dataset_path)}.zip")
    
    # Check if a PPO model already exists and load it if continue_training is True
    model_exists = os.path.exists(ppo_model_path)
    if model_exists and continue_training:
        print(f"Loading existing PPO model from {ppo_model_path}")
        try:
            model = PPO.load(
                ppo_model_path, 
                env=env,
                tensorboard_log=logs_dir,
                custom_objects={"learning_rate": 3e-4}  # Ensure consistent learning rate
            )
            print("Existing PPO model loaded successfully. Continuing training...")
            
            # Update learning rate to potentially lower value for fine-tuning
            model.learning_rate = 1e-4  # Reduced learning rate for continued training
            print(f"Learning rate set to {model.learning_rate} for continued training")
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Creating a new PPO model instead...")
            model_exists = False
    
    # Create a new model if needed
    if not model_exists or not continue_training:
        print("Creating a new PPO agent...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=logs_dir,
            verbose=1
        )
    
    # Create the callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix=f"ppo_certifai_{os.path.basename(dataset_path)}_checkpoint"
    )
    
    monitor_callback = CERTIFAIMonitorCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=10,
        verbose=1
    )
    
    # Display training message
    if model_exists and continue_training:
        print(f"Continuing training for additional {total_timesteps} timesteps...")
    else:
        print(f"Starting new training for {total_timesteps} timesteps...")
    
    # Train the model with callbacks
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, monitor_callback],
        reset_num_timesteps=not (model_exists and continue_training)  # Don't reset if continuing
    )
    
    # Save the final model
    model.save(ppo_model_path)
    print(f"Final model saved to {ppo_model_path}")
    
    # Evaluate the trained policy
    print("Evaluating the trained policy...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model, env

def generate_counterfactuals(ppo_model, env, dataset_path, save_path=None, 
                            specific_indices=None, max_steps_per_sample=100,
                            batch_size=None):
    """
    Generate counterfactuals using a trained PPO model for all samples in the dataset.
    
    Parameters:
    -----------
    ppo_model : PPO
        Trained PPO model
    env : CERTIFAIEnv
        Environment for counterfactual generation
    dataset_path : str
        Path to the dataset CSV file
    save_path : str, optional
        Path to save the generated counterfactuals
    specific_indices : list, optional
        List of specific data point indices to generate counterfactuals for
        If None, all samples in the dataset will be used
    max_steps_per_sample : int
        Maximum steps to attempt for each counterfactual
    batch_size : int, optional
        Number of samples to process before saving intermediate results.
        If None, all results will be saved only at the end.
        
    Returns:
    --------
    tuple: (counterfactuals, original_df, counterfactual_df)
        - counterfactuals: List of generated counterfactuals with detailed info
        - original_df: DataFrame of original samples (for KPI calculation)
        - counterfactual_df: DataFrame of counterfactual samples (for KPI calculation)
    """
    print(f"Generating counterfactuals...")
    
    # Load the original dataset for reference
    original_data = pd.read_csv(dataset_path)
    
    # Set up indices to use
    if specific_indices is not None:
        print(f"Using {len(specific_indices)} specific indices from dataset")
        indices_to_use = specific_indices
    else:
        print(f"Generating counterfactuals for ALL samples in the dataset")
        indices_to_use = list(range(len(original_data)))
    
    num_samples = len(indices_to_use)
    print(f"Total samples to process: {num_samples}")
    
    counterfactuals = []
    original_samples = []
    counterfactual_samples = []
    feature_columns = original_data.columns[:-1].tolist()  # Exclude target column
    
    success_count = 0
    total_steps = 0
    start_time = time.time()
    
    for i, idx in enumerate(indices_to_use):
        # Reset environment with specific index
        obs = env.reset()
        env.current_instance_idx = idx
        env.original_features = env.tab_dataset.iloc[env.current_instance_idx].values[:-1]
        env.modified_features = env.original_features.copy()
        env.original_prediction = env.generate_prediction(env.model, env.original_features)
        obs = env._get_observation()
        
        done = False
        steps = 0
        
        original_features = env.original_features
        original_prediction = env.original_prediction
        current_idx = env.current_instance_idx
        
        while not done and steps < max_steps_per_sample:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if info['counterfactual_found']:
                success_count += 1
                total_steps += steps
                
                # Add to counterfactual list (detailed info with metadata)
                counterfactuals.append({
                    'sample_id': i,
                    'data_index': current_idx,
                    'original_features': original_features,
                    'counterfactual_features': info['modified_features'],
                    'original_prediction': original_prediction,
                    'counterfactual_prediction': info['modified_prediction'],
                    'distance': info['distance'],
                    'steps': steps
                })
                
                # Add to dataframes for KPI calculation
                original_samples.append(original_features)
                counterfactual_samples.append(info['modified_features'])
                
                break
        
        # Progress reporting (more frequent for large datasets)
        if (i + 1) % max(1, num_samples // 100) == 0 or i == num_samples - 1:
            elapsed = time.time() - start_time
            time_per_sample = elapsed / (i + 1)
            remaining = time_per_sample * (num_samples - i - 1)
            
            print(f"Progress: {i+1}/{num_samples} samples processed ({(i+1)/num_samples:.1%})")
            print(f"Current success rate: {success_count/(i+1):.2%}")
            print(f"Time elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
        
        # Save intermediate results if batch_size is specified
        if batch_size and (i + 1) % batch_size == 0:
            intermediate_save_path = f"{os.path.splitext(save_path)[0]}_batch_{(i+1)//batch_size}.csv"
            _save_counterfactuals(counterfactuals, original_data, intermediate_save_path)
            print(f"Saved intermediate results to {intermediate_save_path}")
    
    # Create dataframes for KPI calculation
    original_df = pd.DataFrame(original_samples, columns=feature_columns)
    counterfactual_df = pd.DataFrame(counterfactual_samples, columns=feature_columns)
    
    # Final statistics
    success_rate = success_count / num_samples
    mean_distance = np.mean([cf['distance'] for cf in counterfactuals]) if counterfactuals else 0
    avg_steps = total_steps / success_count if success_count > 0 else 0
    print(f"\nFinal statistics:")
    print(f"Success rate: {success_rate:.2%} ({success_count}/{num_samples})")
    print(f"Mean distance of counterfactuals: {mean_distance:.2f}")
    print(f"Average steps per successful counterfactual: {avg_steps:.2f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    # Calculate and print KPIs if successful counterfactuals were found
    if len(counterfactuals) > 0:
        # Determine continuous and categorical features
        continuous_features = []
        categorical_features = []
        
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(original_df[col]):
                continuous_features.append(col)
            else:
                categorical_features.append(col)
        
        try:
            from KPIs import proximity_KPI, sparsity_KPI
            
            # Calculate proximity
            proximity = proximity_KPI(original_df, counterfactual_df, 
                                     con=continuous_features, 
                                     cat=categorical_features)
            print(f"Proximity KPI: {proximity}")
            
            # Calculate sparsity
            sparsity = sparsity_KPI(original_df, counterfactual_df)
            print(f"Sparsity KPI: {sparsity}")
        except ImportError:
            print("KPIs module not found - skipping KPI calculation")
    
    # Save counterfactuals if a save path is provided
    if save_path and counterfactuals:
        _save_counterfactuals(counterfactuals, original_data, save_path)
        
        # Also save the KPI-compatible dataframes
        kpi_base_path = os.path.splitext(save_path)[0]
        original_df.to_csv(f"{kpi_base_path}_original.csv", index=False)
        counterfactual_df.to_csv(f"{kpi_base_path}_counterfactual.csv", index=False)
        print(f"KPI-compatible dataframes saved to {kpi_base_path}_original.csv and {kpi_base_path}_counterfactual.csv")
    
    return counterfactuals, original_df, counterfactual_df 

def _save_counterfactuals(counterfactuals, original_data, save_path):
    """Helper function to save counterfactuals to CSV and generate summary"""
    # Convert to DataFrame for easier analysis
    counterfactual_data = []
    for cf in counterfactuals:
        # Create a row with original and counterfactual features
        row = {
            'sample_id': cf['sample_id'],
            'data_index': cf['data_index'],
            'original_prediction': cf['original_prediction'],
            'counterfactual_prediction': cf['counterfactual_prediction'],
            'distance': cf['distance'],
            'steps': cf['steps']
        }
        
        # Add original features and counterfactual features
        for i, col in enumerate(original_data.columns[:-1]):  # Exclude target column
            row[f'original_{col}'] = cf['original_features'][i]
            row[f'counterfactual_{col}'] = cf['counterfactual_features'][i]
        
        counterfactual_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(counterfactual_data)
    df.to_csv(save_path, index=False)
    print(f"Counterfactuals saved to {save_path}")
    
    # Create a summary file with feature change frequencies
    if len(counterfactual_data) > 0:
        summary_path = os.path.splitext(save_path)[0] + "_summary.csv"
        
        # Calculate which features changed most often
        feature_changes = {}
        for i, col in enumerate(original_data.columns[:-1]):
            changes = 0
            total_abs_change = 0
            
            for cf in counterfactuals:
                orig = cf['original_features'][i]
                new = cf['counterfactual_features'][i]
                
                if abs(orig - new) > 1e-6:  # Consider it changed if difference is significant
                    changes += 1
                    total_abs_change += abs(orig - new)
            
            feature_changes[col] = {
                'change_count': changes,
                'change_percentage': changes / len(counterfactuals) * 100 if counterfactuals else 0,
                'avg_magnitude': total_abs_change / changes if changes > 0 else 0
            }
        
        # Create summary DataFrame
        summary_df = pd.DataFrame.from_dict(feature_changes, orient='index')
        summary_df.sort_values('change_percentage', ascending=False, inplace=True)
        summary_df.to_csv(summary_path)
        print(f"Feature change summary saved to {summary_path}")
    
    return counterfactuals

TOTAL_TIMESTEPS = 200000  # Total timesteps for training

def main():
    # Specify the dataset path
    dataset_path = 'diabetes.csv'  # Change to your dataset
    
    # Create logs and model directories
    logs_dir = 'ppo_logs'
    save_dir = 'ppo_models'
    
    # Define whether to continue training an existing model
    continue_training = True
    
    # Train the PPO model (will load and continue if it exists)
    ppo_model, env = train_ppo_for_counterfactuals(
        dataset_path=dataset_path,
        logs_dir=logs_dir,
        save_dir=save_dir,
        total_timesteps=TOTAL_TIMESTEPS, 
        continue_training=continue_training
    )
    
    if ppo_model is not None:
        # Generate counterfactuals
        counterfactuals = generate_counterfactuals(
            ppo_model=ppo_model,
            env=env,
            dataset_path=dataset_path,
            #num_samples=20,
            save_path=f'generated_counterfactuals_{os.path.splitext(os.path.basename(dataset_path))[0]}.csv'
        )
        
        print(f"Generated {len(counterfactuals[0])} counterfactuals successfully")

if __name__ == "__main__":
    main()