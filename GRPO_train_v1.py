import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.distributions import Categorical
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from GRPO_env import GRPOEnv, GRPOMonitorCallback
from Classifier_model import Classifier, train_model
from KPIs import proximity_KPI, sparsity_KPI


class GRPOPolicy(nn.Module):
    """
    Simple policy network for GRPO - no critic needed!
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(GRPOPolicy, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs):
        """Forward pass through the policy network"""
        logits = self.network(obs)
        return logits
    
    def get_action_and_log_prob(self, obs, deterministic=False):
        """Get action and log probability for given observation"""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def get_log_prob(self, obs, action):
        """Get log probability for given observation and action"""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)


class GRPOTrainer:
    """
    GRPO Trainer for counterfactual generation
    """
    def __init__(self, env, policy, learning_rate=3e-4, clip_range=0.2, 
                 kl_coef=0.01, group_size=4, max_grad_norm=0.5):
        self.env = env
        self.policy = policy
        self.group_size = group_size
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Reference policy for KL penalty (frozen copy of initial policy)
        self.reference_policy = GRPOPolicy(
            obs_dim=policy.network[0].in_features,
            action_dim=policy.network[-1].out_features,
            hidden_dim=256
        )
        self.reference_policy.load_state_dict(policy.state_dict())
        self.reference_policy.eval()
        
        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'kl_divergence': [],
            'success_rate': [],
            'mean_reward': [],
            'mean_advantage': []
        }
    
    def compute_grpo_loss(self, trajectories):
        """
        Compute GRPO loss from group trajectories
        """
        total_loss = 0
        total_kl = 0
        num_steps = 0
        
        # Compute group advantages
        advantages = self.env.compute_group_advantages(trajectories)
        
        for traj_idx, (trajectory, advantage) in enumerate(zip(trajectories, advantages)):
            if len(trajectory['states']) == 0:
                continue
                
            # Convert to tensors
            states = torch.FloatTensor(np.array(trajectory['states']))
            actions = torch.LongTensor(trajectory['actions'])
            
            # Get current policy log probabilities
            current_log_probs = self.policy.get_log_prob(states, actions)
            
            # Get reference policy log probabilities for KL penalty
            with torch.no_grad():
                ref_log_probs = self.reference_policy.get_log_prob(states, actions)
            
            # Compute probability ratios (similar to PPO)
            if len(trajectory['log_probs']) > 0:
                old_log_probs = torch.FloatTensor(trajectory['log_probs'])
            else:
                # Fallback if log_probs weren't stored properly
                old_log_probs = ref_log_probs.clone()
            
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # GRPO clipped surrogate loss (same as PPO but with group advantages)
            advantage_tensor = torch.tensor(advantage, dtype=torch.float32)
            surrogate1 = ratio * advantage_tensor
            surrogate2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage_tensor
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # KL divergence penalty
            kl_div = (current_log_probs - ref_log_probs).mean()
            
            total_loss += policy_loss + self.kl_coef * kl_div
            total_kl += kl_div.item()
            num_steps += len(trajectory['states'])
        
        return total_loss / len(trajectories), total_kl / len(trajectories)
    
    def train_step(self, num_episodes=10):
        """
        Perform one training step with group sampling
        """
        all_trajectories = []
        episode_rewards = []
        success_count = 0
        
        # Sample multiple groups of trajectories
        for episode in range(num_episodes):
            # Sample group trajectories for the same initial state
            trajectories = self.env.sample_group_trajectories(
                policy_fn=self.policy_wrapper,
                group_size=self.group_size
            )
            
            all_trajectories.extend(trajectories)
            
            # Track statistics
            for traj in trajectories:
                episode_rewards.append(traj['total_reward'])
                if traj['counterfactual_found']:
                    success_count += 1
        
        # Compute GRPO loss
        policy_loss, kl_div = self.compute_grpo_loss(all_trajectories)
        
        # Optimize policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update training stats
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['kl_divergence'].append(kl_div)
        self.training_stats['success_rate'].append(success_count / len(all_trajectories))
        self.training_stats['mean_reward'].append(np.mean(episode_rewards))
        
        # Compute mean advantage for logging
        if all_trajectories:
            advantages = self.env.compute_group_advantages(all_trajectories)
            self.training_stats['mean_advantage'].append(np.mean(advantages))
        
        return {
            'policy_loss': policy_loss.item(),
            'kl_divergence': kl_div,
            'success_rate': success_count / len(all_trajectories),
            'mean_reward': np.mean(episode_rewards),
            'total_trajectories': len(all_trajectories)
        }
    
    def policy_wrapper(self, obs):
        """Wrapper for policy to work with environment sampling"""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        action, log_prob = self.policy.get_action_and_log_prob(obs.unsqueeze(0))
        return action.item(), log_prob.item()
    
    def update_reference_policy(self):
        """Update reference policy (for KL penalty) - called periodically"""
        self.reference_policy.load_state_dict(self.policy.state_dict())
        self.reference_policy.eval()


def train_grpo_for_counterfactuals(dataset_path, model_path=None, logs_dir='grpo_logs', 
                                   save_dir='grpo_models', total_steps=1000, 
                                   continue_training=True):
    """
    Train a GRPO agent to generate counterfactuals for a given classifier model.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset CSV file
    model_path : str, optional
        Path to the pre-trained classifier model
    logs_dir : str
        Directory for storing logs
    save_dir : str
        Directory for saving GRPO model checkpoints
    total_steps : int
        Total number of training steps
    continue_training : bool
        Whether to continue training an existing GRPO model if available
        
    Returns:
    --------
    trainer : GRPOTrainer
        Trained GRPO trainer
    env : GRPOEnv
        Environment used for training
    """
    print(f"Starting counterfactual generation with GRPO for dataset: {dataset_path}")
    
    # Create directories if they don't exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load dataset and determine output size
    dataset = pd.read_csv(dataset_path)
    out = len(dataset.iloc[:, -1].unique())
    in_feats = len(dataset.columns) - 1  # Exclude target variable
    
    # Load or train the classifier model
    if model_path is None:
        model_path = f"{os.path.splitext(os.path.basename(dataset_path))[0]}_model.pt"
        model_path = os.path.join("classification_models", model_path)
    
    if not os.path.exists(model_path):
        print(f"Classifier model not found at {model_path}. Training a new one.")
        train_model(dataset_path, model_path)

    print(f"Loading classifier model from {model_path}")
    
    # Load the classifier model
    classifier = Classifier(in_feats=in_feats, out=out)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()  # Set to evaluation mode
    
    # Create the GRPO environment
    env = GRPOEnv(dataset_path=dataset_path, model=classifier)
    
    # Create policy network
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = GRPOPolicy(obs_dim=obs_dim, action_dim=action_dim)
    
    # Define model paths
    policy_model_path = os.path.join(save_dir, f"grpo_policy_{os.path.basename(dataset_path)}.pt")
    
    # Check if existing model should be loaded
    model_exists = os.path.exists(policy_model_path)
    if model_exists and continue_training:
        print(f"Loading existing GRPO policy from {policy_model_path}")
        try:
            policy.load_state_dict(torch.load(policy_model_path))
            print("Existing GRPO policy loaded successfully. Continuing training...")
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Creating a new GRPO policy instead...")
            model_exists = False
    
    # Create GRPO trainer
    trainer = GRPOTrainer(
        env=env,
        policy=policy,
        learning_rate=3e-4,
        group_size=4,  # Number of trajectories to compare
        clip_range=0.2,
        kl_coef=0.01
    )
    
    # Training loop
    print(f"Starting GRPO training for {total_steps} steps...")
    start_time = time.time()
    
    for step in tqdm(range(total_steps), desc="Training GRPO"):
        # Train step
        step_stats = trainer.train_step(num_episodes=5)  # 5 episodes per step
        
        # Log progress
        if step % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"\nStep {step}/{total_steps}")
            print(f"Policy Loss: {step_stats['policy_loss']:.4f}")
            print(f"Success Rate: {step_stats['success_rate']:.2%}")
            print(f"Mean Reward: {step_stats['mean_reward']:.2f}")
            print(f"KL Divergence: {step_stats['kl_divergence']:.4f}")
            print(f"Time elapsed: {elapsed_time:.1f}s")
        
        # Update reference policy periodically
        if step % 200 == 0:
            trainer.update_reference_policy()
        
        # Save checkpoint
        if step % 500 == 0:
            checkpoint_path = os.path.join(save_dir, f"grpo_policy_checkpoint_{step}_{os.path.basename(dataset_path)}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
    
    # Save final model
    torch.save(policy.state_dict(), policy_model_path)
    print(f"Final GRPO policy saved to {policy_model_path}")
    
    return trainer, env


def generate_counterfactuals_grpo(trainer, env, dataset_path, save_path=None, 
                                  specific_indices=None, max_steps_per_sample=100):
    """
    Generate counterfactuals using a trained GRPO policy.
    
    Parameters:
    -----------
    trainer : GRPOTrainer
        Trained GRPO trainer containing the policy
    env : GRPOEnv
        Environment for counterfactual generation
    dataset_path : str
        Path to the dataset CSV file
    save_path : str, optional
        Path to save the generated counterfactuals
    specific_indices : list, optional
        List of specific data point indices to generate counterfactuals for
    max_steps_per_sample : int
        Maximum steps to attempt for each counterfactual
        
    Returns:
    --------
    tuple: (counterfactuals, original_df, counterfactual_df)
    """
    print(f"Generating counterfactuals using GRPO...")
    
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
    
    # Set policy to evaluation mode
    trainer.policy.eval()
    
    for i, idx in tqdm(enumerate(indices_to_use), total=num_samples, desc="Generating counterfactuals"):
        # Reset environment with specific index
        obs = env.reset(instance_idx=idx)
        
        done = False
        steps = 0
        
        original_features = env.original_features
        original_prediction = env.original_prediction
        current_idx = env.current_instance_idx
        
        while not done and steps < max_steps_per_sample:
            # Get action from GRPO policy (deterministic for evaluation)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _ = trainer.policy.get_action_and_log_prob(obs_tensor, deterministic=True)
                action = action.item()
                
            obs, reward, done, info = env.step(action)
            steps += 1
            
            # If a counterfactual is found, record it
            if info['counterfactual_found']:
                success_count += 1
                total_steps += steps
                
                # Add to counterfactual list
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
                
                print(f"Sample {i}: distance={info['distance']:.3f}, steps={steps}")
                
                # Add to dataframes for KPI calculation
                original_samples.append(original_features)
                counterfactual_samples.append(info['modified_features'])
                
                break
    
    # Create dataframes for KPI calculation
    original_df = pd.DataFrame(original_samples, columns=feature_columns)
    counterfactual_df = pd.DataFrame(counterfactual_samples, columns=feature_columns)
    
    # Final statistics
    success_rate = success_count / num_samples
    mean_distance = np.mean([cf['distance'] for cf in counterfactuals]) if counterfactuals else 0
    avg_steps = total_steps / success_count if success_count > 0 else 0
    
    print(f"\nGRPO Final statistics:")
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
    
    # Save counterfactuals if path provided
    if save_path and counterfactuals:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        _save_counterfactuals_grpo(counterfactuals, original_data, save_path)
        
        # Also save KPI-compatible dataframes
        kpi_base_path = os.path.splitext(save_path)[0]
        original_df.to_csv(f"{kpi_base_path}_original.csv", index=False)
        counterfactual_df.to_csv(f"{kpi_base_path}_counterfactual.csv", index=False)
    
    return counterfactuals, original_df, counterfactual_df


def _save_counterfactuals_grpo(counterfactuals, original_data, save_path):
    """Helper function to save GRPO counterfactuals to CSV"""
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
    print(f"GRPO counterfactuals saved to {save_path}")


def main():
    """Main function to run GRPO training and counterfactual generation"""
    dataset_path = 'data/adult.csv'
    dataset_path = 'data/diabetes.csv'
    
    model_path = None  # Will be auto-determined
    logs_dir = 'grpo_logs'
    save_dir = 'grpo_models'
    os.makedirs('data', exist_ok=True)
    
    # Training settings
    total_training_steps = 500  # Adjust as needed
    continue_training = True
    
    print("=== GRPO Counterfactual Generation ===")
    
    # Train GRPO model
    trainer, env = train_grpo_for_counterfactuals(
        dataset_path=dataset_path,
        logs_dir=logs_dir,
        save_dir=save_dir,
        total_steps=total_training_steps,
        continue_training=continue_training
    )
    
    # Generate counterfactuals using trained GRPO
    indices_to_use = list(range(100))  # First 100 samples
    
    counterfactuals, _, _ = generate_counterfactuals_grpo(
        trainer=trainer,
        env=env,
        dataset_path=dataset_path,
        save_path=f'data/generated_counterfactuals_grpo_{os.path.splitext(os.path.basename(dataset_path))[0]}.csv',
        max_steps_per_sample=100,
        specific_indices=indices_to_use
    )
    
    print(f"\nGRPO training and counterfactual generation completed!")
    print(f"Generated {len(counterfactuals)} counterfactuals")


if __name__ == "__main__":
    main()