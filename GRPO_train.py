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


"""
The following code implements the GRPO algorithm for training a policy to generate counterfactuals.
This is based on the Deepseek GRPO algorithm, as described in their paper.

Algorithm 1 Iterative Group Relative Policy Optimization
Input initial policy model 𝜋𝜃init ; reward models 𝑟𝜑; task prompts D; hyperparameters 𝜀, 𝛽, 𝜇
1: policy model 𝜋𝜃 ← 𝜋𝜃init
2: for iteration = 1, . . . , I do
3: reference model 𝜋𝑟𝑒 𝑓 ← 𝜋𝜃
4: for step = 1, . . . , M do
5: Sample a batch D𝑏 from D
6: Update the old policy model 𝜋𝜃𝑜𝑙𝑑 ← 𝜋𝜃
7: Sample 𝐺 outputs {𝑜𝑖}𝐺𝑖=1 ∼ 𝜋𝜃𝑜𝑙𝑑 (· | 𝑞) for each question 𝑞 ∈ D𝑏
8: Compute rewards {𝑟𝑖}𝐺𝑖=1 for each sampled output 𝑜𝑖 by running 𝑟𝜑
9: Compute ˆ𝐴𝑖,𝑡 for the 𝑡-th token of 𝑜𝑖 through group relative advantage estimation.
10: for GRPO iteration = 1, . . . , 𝜇 do
11: Update the policy model 𝜋𝜃 by maximizing the GRPO objective (Equation 21)
12: Update 𝑟𝜑 through continuous training using a replay mechanism.
Output 𝜋𝜃

"""

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
    GRPO Trainer following Deepseek GRPO Algorithm
    """
    def __init__(self, env, policy, learning_rate=1e-3, clip_range=0.2, 
                 kl_coef=0.01, group_size=8, max_grad_norm=0.5, mu_iterations=4):
        self.env = env
        self.policy = policy
        self.group_size = group_size
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.mu_iterations = mu_iterations  # Inner GRPO iterations (μ in algorithm)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Reference policy for KL penalty (π_ref in algorithm)
        self.reference_policy = self._create_policy_copy()
        self.reference_policy.eval()
        
        # Old policy for importance sampling (π_θ_old in algorithm)
        self.old_policy = None
        
        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'kl_divergence': [],
            'success_rate': [],
            'mean_reward': [],
            'mean_advantage': []
        }
    
    def _create_policy_copy(self):
        """Create a deep copy of the current policy"""
        policy_copy = GRPOPolicy(
            obs_dim=self.policy.network[0].in_features,
            action_dim=self.policy.network[-1].out_features,
            hidden_dim=256
        )
        policy_copy.load_state_dict(self.policy.state_dict())
        return policy_copy
    
    def train_step(self, num_episodes=10):
        """
        Perform one training step following Deepseek GRPO Algorithm structure
        """
        # Step 6: Update old policy model π_θ_old ← π_θ
        self.old_policy = self._create_policy_copy()
        self.old_policy.eval()
        
        all_trajectories = []
        episode_rewards = []
        success_count = 0
        
        # Steps 5-8: Sample batch and generate trajectories with rewards
        for episode in range(num_episodes):
            # Step 7: Sample G outputs from π_θ_old
            trajectories = self._sample_group_trajectories_with_old_policy(
                group_size=self.group_size
            )
            
            all_trajectories.extend(trajectories)
            
            # Track statistics
            for traj in trajectories:
                episode_rewards.append(traj['total_reward'])
                if traj['counterfactual_found']:
                    success_count += 1
        
        # Step 9: Compute group relative advantages
        advantages = self.env.compute_group_advantages(all_trajectories)
        
        # Steps 10-11: Inner GRPO iterations (μ times)
        total_policy_loss = 0
        total_kl_div = 0
        
        for _ in range(self.mu_iterations):
            policy_loss, kl_div = self._compute_grpo_loss(all_trajectories, advantages)
            
            # Optimize policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_kl_div += kl_div
        
        # Average losses over inner iterations
        avg_policy_loss = total_policy_loss / self.mu_iterations
        avg_kl_div = total_kl_div / self.mu_iterations
        
        # Update training stats
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['kl_divergence'].append(avg_kl_div)
        self.training_stats['success_rate'].append(success_count / len(all_trajectories))
        self.training_stats['mean_reward'].append(np.mean(episode_rewards))
        
        if all_trajectories:
            self.training_stats['mean_advantage'].append(np.mean(advantages))
        
        return {
            'policy_loss': avg_policy_loss,
            'kl_divergence': avg_kl_div,
            'success_rate': success_count / len(all_trajectories),
            'mean_reward': np.mean(episode_rewards),
            'total_trajectories': len(all_trajectories),
            'grpo_iterations': self.mu_iterations
        }
    
    def _sample_group_trajectories_with_old_policy(self, group_size):
        """
        Sample trajectories using the old policy (π_θ_old)
        This ensures we use the policy that was active when we started this batch
        """
        trajectories = []
        # Reset to same initial state for group comparison
        self.env.reset()
        initial_idx = self.env.current_instance_idx
        
        for _ in range(group_size):
            # Reset to same initial state 
            self.env.reset(instance_idx=initial_idx)
            
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'done': False,
                'total_reward': 0,
                'counterfactual_found': False
            }
            
            obs = self.env._get_observation()
            steps = 0
            
            while not self.env.done and steps < self.env.max_steps:
                # Sample action from OLD policy (π_θ_old)
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action, log_prob = self.old_policy.get_action_and_log_prob(obs_tensor)
                    action = action.item()
                    log_prob = log_prob.item()
                
                trajectory['states'].append(obs.copy())
                trajectory['actions'].append(action)
                trajectory['log_probs'].append(log_prob)
                
                obs, reward, done, info = self.env.step(action)
                
                trajectory['rewards'].append(reward)
                trajectory['total_reward'] += reward
                
                if info['counterfactual_found']:
                    trajectory['counterfactual_found'] = True
                    trajectory['distance'] = info['distance']
                    trajectory['steps_to_success'] = steps + 1
                
                steps += 1
            
            trajectory['done'] = self.env.done
            trajectory['final_info'] = info if 'info' in locals() else {}
            trajectories.append(trajectory)
        
        return trajectories
    
    def _compute_grpo_loss(self, trajectories, advantages):
        """
        Compute GRPO loss using current policy and stored trajectories/advantages
        """
        total_loss = 0
        total_kl = 0
        num_trajectories = 0
        
        for traj_idx, (trajectory, advantage) in enumerate(zip(trajectories, advantages)):
            if len(trajectory['states']) == 0:
                continue
                
            # Convert to tensors
            states = torch.FloatTensor(np.array(trajectory['states']))
            actions = torch.LongTensor(trajectory['actions'])
            old_log_probs = torch.FloatTensor(trajectory['log_probs'])
            
            # Get current policy log probabilities
            current_log_probs = self.policy.get_log_prob(states, actions)
            
            # Get reference policy log probabilities for KL penalty
            with torch.no_grad():
                ref_log_probs = self.reference_policy.get_log_prob(states, actions)
            
            # Compute importance sampling ratio
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # GRPO clipped surrogate loss with group advantages
            advantage_tensor = torch.tensor(advantage, dtype=torch.float32)
            surrogate1 = ratio * advantage_tensor
            surrogate2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage_tensor
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # KL divergence penalty
            kl_div = (current_log_probs - ref_log_probs).mean()
            
            total_loss += policy_loss + self.kl_coef * kl_div
            total_kl += kl_div.item()
            num_trajectories += 1
        
        if num_trajectories == 0:
            return torch.tensor(0.0, requires_grad=True), 0.0
            
        return total_loss / num_trajectories, total_kl / num_trajectories
    
    def update_reference_policy(self):
        """Step 3: Update reference policy π_ref ← π_θ"""
        self.reference_policy.load_state_dict(self.policy.state_dict())
        self.reference_policy.eval()
    
    def policy_wrapper(self, obs):
        """Wrapper for policy to work with environment sampling"""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        action, log_prob = self.policy.get_action_and_log_prob(obs.unsqueeze(0))
        return action.item(), log_prob.item()

def train_grpo_for_counterfactuals(dataset_path, model_path=None, logs_dir='grpo_logs', 
                                   save_dir='grpo_models', total_iterations=100, 
                                   steps_per_iteration=10, mu_iterations=4,
                                   continue_training=True):
    """
    Train GRPO following Deepseek GRPO Algorithm structure
    
    Parameters:
    -----------
    total_iterations : int
        Number of outer iterations (I in algorithm)
    steps_per_iteration : int  
        Number of steps per iteration (M in algorithm)
    mu_iterations : int
        Number of inner GRPO iterations (μ in algorithm)
    """
    print(f"Starting GRPO training with Deepseek GRPO Algorithm structure")
    print(f"Iterations: {total_iterations}, Steps per iteration: {steps_per_iteration}, μ iterations: {mu_iterations}")
    
    
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
    
     # Create GRPO trainer with μ parameter
    trainer = GRPOTrainer(
        env=env,
        policy=policy,
        learning_rate=3e-4,
        group_size=4,
        clip_range=0.2,
        kl_coef=0.01,
        mu_iterations=mu_iterations
    )
    
    # Deepseek GRPO Algorithm: Main training loop
    print(f"Starting GRPO training following Deepseek GRPO Algorithm...")
    
    for iteration in range(total_iterations):  # Algorithm line 2
        # Step 3: Update reference model
        trainer.update_reference_policy()
        
        print(f"\n=== Iteration {iteration+1}/{total_iterations} ===")
        
        # Steps 4-11: Inner training loop
        for step in range(steps_per_iteration):
            step_stats = trainer.train_step(num_episodes=5)
            
            if step % 1 == 0:
                print(f"  Step {step+1}/{steps_per_iteration}: "
                      f"Loss={step_stats['policy_loss']:.4f}, "
                      f"Success={step_stats['success_rate']:.2%}, "
                      f"μ_iters={step_stats['grpo_iterations']}")
        
        # Log iteration statistics
        recent_stats = trainer.training_stats
        if recent_stats['policy_loss']:
            avg_loss = np.mean(recent_stats['policy_loss'][-steps_per_iteration:])
            avg_success = np.mean(recent_stats['success_rate'][-steps_per_iteration:])
            avg_reward = np.mean(recent_stats['mean_reward'][-steps_per_iteration:])
            
            print(f"Iteration {iteration+1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Success Rate: {avg_success:.2%}")
            print(f"  Average Reward: {avg_reward:.2f}")
        
        # Save checkpoint every few iterations
        if (iteration + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"grpo_policy_iter_{iteration+1}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(save_dir, f"grpo_policy_final.pt")
    torch.save(policy.state_dict(), final_path)
    print(f"Final model saved: {final_path}")
    
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
    
    print("=== GRPO Counterfactual Generation ===")
    
    # Train GRPO model
    trainer, env = train_grpo_for_counterfactuals(
        dataset_path='data/diabetes.csv',
        total_iterations=10,      # I   
        steps_per_iteration=10,   # M 
        mu_iterations=4,          # μ 
        continue_training=False
    )
    
    # Generate counterfactuals using trained GRPO
    indices_to_use = list(range(100)) 
    
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