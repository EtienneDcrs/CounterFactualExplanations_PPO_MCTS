# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import manhattan_distances as L1
from sklearn.metrics.pairwise import euclidean_distances as L2
from skimage.metrics import structural_similarity as SSIM
from tqdm import tqdm
from KPIs import proximity_KPI, sparsity_KPI, validity_KPI
warnings.filterwarnings("ignore")

class CERTIFAI_PPO:
    def __init__(self, dataset_path = None, numpy_dataset = None):
    
        self.dataset_path = dataset_path
        self.numpy_dataset = numpy_dataset
        self.tab_dataset = None
        self.predictions = None
        self.distance = L2
        self.state_space = None
        self.action_space = None
        self.constraints = None 
        self.constraints = [1,1,1,1,1]
        self.constraints = [1,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0]
        self.policy_network = None
        self.value_network = None
        self.optimizer = None

        self.generation = 0
        self.max_generations = 20

        if dataset_path is not None:
            self.tab_dataset = pd.read_csv(dataset_path)
        elif numpy_dataset is not None:
            self.tab_dataset = numpy_dataset

        self.results = [None] * len(self.tab_dataset) if self.tab_dataset is not None else []
        self.best_distances = [float('inf')] * len(self.tab_dataset) if self.tab_dataset is not None else []
        

        self.initialize_state_action_spaces()
        self.initialize_ppo_networks()
        self.cats_ids = self.generate_cats_ids(self.tab_dataset)

        self.rewards_history = []
        self.policy_loss_history = []
        self.value_loss_history = []

    def initialize_state_action_spaces(self):
        # Define the state space: original and modified features
        if self.tab_dataset is not None:
            self.state_space = self.tab_dataset.columns.tolist() * 2  # Original and modified features
        # Define the action space: possible modifications to features
        self.action_space = self.define_action_space()

    def define_action_space(self, constraints=None):
        # Example action space definition
        action_space = []
        if constraints is None:
            constraints = self.constraints  # Use default constraints if not provided
        if constraints is None:
            constraints = [1] * len(self.tab_dataset.columns)  # Default to no constraints

        for i, column in enumerate(self.tab_dataset.columns[:-1]):  # Exclude the target variable
            # if the feature is constrained, skip it : fixed features aren't present in the action space
            if constraints[i] == 0:
                    continue
            if self.tab_dataset[column].dtype == 'O':  # Categorical feature
                action_space.append(f'change_{column}')
            else:  # Continuous feature
                action_space.append(f'increase_{column}')
                action_space.append(f'decrease_{column}')
        #print("Action space defined:", action_space)
        self.actions_stats = [0] * len(action_space)
        return action_space

    def initialize_ppo_networks(self):
        # Calculate the correct input dimension
        input_dim = len(self.tab_dataset.columns) * 2 - 2  # Original and modified features minus target variable (x2)
        output_dim = len(self.action_space)

        # Initialize policy and value networks with the correct input dimension
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Initialize with smaller learning rate for stability
        self.optimizer = optim.Adam(list(self.policy_network.parameters()) + list(self.value_network.parameters()), lr=0.0005)

    def get_state(self, original_features, modified_features):
        # Ensure original_features and modified_features are correctly shaped
        if original_features.ndim == 1:
            original_features = original_features.reshape(1, -1)
        if modified_features.ndim == 1:
            modified_features = modified_features.reshape(1, -1)

        # Ensure features are DataFrames
        original_features = pd.DataFrame(original_features, columns=self.tab_dataset.columns)
        modified_features = pd.DataFrame(modified_features, columns=self.tab_dataset.columns)

        # Encode categorical features
        original_features = self.encode_features(original_features)
        modified_features = self.encode_features(modified_features)

        # Combine original and modified features to form the state
        state = np.concatenate([original_features, modified_features], axis=-1)
        
        # Check for NaN or inf values and replace them
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return state

    def encode_features(self, features):
        cat_ids = self.generate_cats_ids(self.tab_dataset)
        
        # Ensure features is a DataFrame
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features.reshape(1, -1), columns=self.tab_dataset.columns)
        encoded_features = features.copy()    
        # Convert categorical values to their indices
        for idx, n_cats, cat_values in cat_ids:
            for i, cat_value in enumerate(cat_values):
                # Replace categorical values with their index
                encoded_features.iloc[:, idx] = np.where(encoded_features.iloc[:, idx] == cat_value, i, encoded_features.iloc[:, idx])

        # Convert to float32 and handle any NaN values
        encoded = encoded_features.astype(np.float32).values
        encoded = np.nan_to_num(encoded, nan=0.0)
        return encoded

    def decode_features(self, features):
        # Decode categorical features back to their original values
        cat_ids = self.generate_cats_ids(self.tab_dataset)

        # Ensure features is a DataFrame
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features.reshape(1, -1))

        decoded_features = features.copy()

        # Convert indices back to categorical values
        for idx, n_cats, cat_values in cat_ids:
            # Ensure the column is of object type to handle categorical values
            decoded_features.iloc[:, idx] = decoded_features.iloc[:, idx].astype(object)
            for i, cat_value in enumerate(cat_values):
                # Replace indices with their original categorical values
                condition = decoded_features.iloc[:, idx] == i
                decoded_features.loc[condition, decoded_features.columns[idx]] = cat_value

        output = decoded_features.to_numpy(dtype=object)
        return output

    def select_action(self, state):
        """
        Select an action based on the policy network with adaptive exploration.
        
        Parameters:
        - state: Current state representation
        
        Returns:
        - action: Selected action index
        """
        with torch.no_grad():
            tensor_state = torch.tensor(state, dtype=torch.float32)
            
            # Check for NaN values
            if torch.isnan(tensor_state).any():
                tensor_state = torch.nan_to_num(tensor_state, nan=0.0)
            
            # Get action probabilities from policy network
            action_probs = self.policy_network(tensor_state)
            
            # Handle any issues with probabilities
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                # Reset to uniform distribution if problems detected
                action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
            
            # Ensure minimum probability for all actions (prevent zeros)
            min_prob = 1e-6
            action_probs = torch.clamp(action_probs, min=min_prob)
            
            # Adaptive exploration rate based on training progress
            # More exploration early, more exploitation later
            generation = getattr(self, 'generation', 0)
            max_generations = getattr(self, 'max_generations', 20)
            progress = min(generation / max_generations, 1.0) if max_generations > 0 else 0.5
            
            # Exploration schedule: start high, gradually decrease
            init_exploration = 0.3  # 30% exploration at start
            final_exploration = 0.05  # 5% exploration at end
            exploration_rate = init_exploration * (1 - progress) + final_exploration * progress
            
            # Apply exploration noise
            noise = torch.rand_like(action_probs) * exploration_rate
            action_probs = action_probs * (1 - exploration_rate) + noise
            
            # Renormalize to ensure they sum to 1
            action_probs = action_probs / action_probs.sum()
            
            # Sample action based on probabilities
            try:
                action = torch.multinomial(action_probs, 1).item()
            except RuntimeError as e:
                print(f"Error in action sampling: {e}")
                # Fallback to argmax as a last resort
                action = torch.argmax(action_probs).item()
        
        return action

    def apply_action(self, state, action, cat_ids=None):
        # Apply the selected action to the state
        action_name = self.action_space[action]
        self.actions_stats[action] += 1
        modified_features = state[0, len(self.tab_dataset.columns):].copy()
        
        try:
            # Extract feature name and index
            if '_' in action_name:
                feature_name = '_'.join(action_name.split('_')[1:])
                feature_index = self.tab_dataset.columns.get_loc(feature_name)
                
                # Handle categorical features
                if 'change' in action_name:
                    for idx, ncat, cat_value in cat_ids:
                        # Skip if not applicable
                        if ncat <= 1 or feature_index != idx:
                            continue
                        
                        # Current value
                        value_index = int(modified_features[feature_index].item())
                        
                        # Instead of random selection, try categories in a more systematic way
                        # Start with categories that are most different from current
                        # This can be domain-specific, but here's a simple approach
                        all_indices = list(range(ncat))
                        all_indices.remove(value_index)  # Remove current index
                        
                        # Sort by some meaningful metric if available (e.g., distance from current)
                        # For simplicity, we just take the first different index
                        if all_indices:
                            modified_features[feature_index] = all_indices[0]
                            
                # Handle continuous features with adaptive step sizes
                elif 'increase' in action_name or 'decrease' in action_name:
                    current_value = modified_features[feature_index].item()
                    
                    # Start with larger steps early in training, smaller steps later
                    generation = getattr(self, 'generation', 0)
                    max_generations = getattr(self, 'max_generations', 20)
                    progress = min(generation / max_generations, 1.0) if max_generations > 0 else 0.5
                    
                    # Adaptive step size: larger at beginning, finer as we progress
                    base_step = 0.15  # 5% change
                    fine_step = 0.05  # 1% change for fine-tuning
                    step_size = base_step * (1 - progress) + fine_step * progress
                    
                    if 'increase' in action_name:
                        max_value = self.tab_dataset[feature_name].max()
                        # Use smaller steps when approaching the boundary
                        calculated_value = current_value * (1 + step_size)
                        # Si le type de la feature est un int, on arrondit
                        if self.tab_dataset[feature_name].dtype == 'int':
                            calculated_value = int(calculated_value)
                        new_value = min(calculated_value, max_value)
                        modified_features[feature_index] = new_value
                    else:  # decrease
                        min_value = self.tab_dataset[feature_name].min()
                        # Use smaller steps when approaching the boundary
                        calculated_value = current_value * (1 - step_size)
                        # Si le type de la feature est un int, on arrondit
                        if self.tab_dataset[feature_name].dtype == 'int':
                            calculated_value = int(calculated_value)
                        new_value = max(calculated_value, min_value)
                        modified_features[feature_index] = new_value
        
        except Exception as e:
            print(f"Error in apply_action: {e}")
            # Return the unmodified features if there's an error
            pass
            
        # Check for NaN or inf values and replace them
        modified_features = np.nan_to_num(modified_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return modified_features

    def calculate_reward(self, original_features, modified_features, model, sample_index):
        try:
            # Get predictions for original and modified instances
            original_prediction = self.generate_prediction(model, original_features)
            modified_prediction = self.generate_prediction(model, modified_features)
            
            # Calculate distance between original and modified features
            distance = self.get_distance(original_features, modified_features)
            best_distance = self.best_distances[sample_index]
            
            # Handle invalid distances
            if np.isnan(distance) or np.isinf(distance):
                return -100.0  # Large penalty for invalid states
            
            # Calculate the number of modified features (for sparsity)
            modified_features_count = np.sum(self.encode_features(original_features) != 
                                            self.encode_features(modified_features))
            
            # If we've found a counterfactual (class changed)
            if original_prediction != modified_prediction:
                print(f"Found counterfactual for sample {sample_index}: {original_features} -> {modified_features}")
                print(f"Distance: {distance}, Modified features count: {modified_features_count}")
                # Higher reward for successful counterfactual with smaller distance
                base_reward = 100.0
                # Penalize based on distance
                distance_penalty = min(distance * 5, 50)  # Cap the penalty at 50
                # Penalize for modifying too many features
                sparsity_penalty = min(modified_features_count * 2, 30)  # Cap at 30
                
                # Calculate final reward
                reward = base_reward - distance_penalty - sparsity_penalty
                
                # Bonus for improving upon the best distance found so far
                if distance < best_distance:
                    reward += 50.0
                    self.best_distances[sample_index] = distance  # Update best distance
                
                return max(reward, 10.0)  # Ensure minimum positive reward for success
            else:
                # For unsuccessful counterfactuals, provide gradient through probability
                with torch.no_grad():
                    # Encode features for model input
                    original_features_tensor = torch.tensor(self.encode_features(original_features), 
                                                        dtype=torch.float32)
                    modified_features_tensor = torch.tensor(self.encode_features(modified_features), 
                                                        dtype=torch.float32)
                    
                    # Handle potential NaN values
                    original_features_tensor = torch.nan_to_num(original_features_tensor)
                    modified_features_tensor = torch.nan_to_num(modified_features_tensor)
                    
                    # Get model outputs
                    original_logits = model(original_features_tensor)
                    modified_logits = model(modified_features_tensor)
                    
                    # Convert to probabilities
                    original_probs = torch.softmax(original_logits, dim=1)
                    modified_probs = torch.softmax(modified_logits, dim=1)
                    
                    # Target class - the class we want to change to
                    target_class = 1 - original_prediction  # Assuming binary classification
                    
                    # Calculate probability shift toward target class
                    prob_shift = (modified_probs[0, target_class] - 
                                original_probs[0, target_class]).item()
                    
                    # Reward based on probability shift, penalize for distance
                    prob_reward = prob_shift * 75  # Scale up the probability shift reward
                    distance_penalty = min(distance, 25)  # Cap distance penalty
                    
                    # Total reward for unsuccessful attempt
                    reward = prob_reward - distance_penalty - (modified_features_count * 1.5)
                    
                    # Give slightly better rewards for getting closer to success
                    if modified_prediction == target_class:
                        reward += 5.0
                    
                    # Ensure the reward is at least -50 to prevent extreme penalties
                    return max(reward, -50.0)
        
        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            return -50.0  # Return a significant penalty if there's an error
        
    def get_distance(self, original_features, modified_features):
        try:
            features_diff = self.encode_features(original_features) - self.encode_features(modified_features)
            # Replace any NaN values
            features_diff = np.nan_to_num(features_diff, nan=0.0)
            distance = np.linalg.norm(features_diff, ord=2)  # Euclidean distance
            return distance
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return float('inf')  # Return a large distance in case of error

    def generate_prediction(self, model, features):
        try:
            # Ensure features are numerical
            features = self.encode_features(features)
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model(features_tensor).argmax(dim=-1).item()
            return prediction
        except Exception as e:
            print(f"Error in generate_prediction: {e}")
            return 0  # Return a default prediction in case of error

    def generate_cats_ids(self, dataset = None, cat = None):
        if dataset is None:
            assert self.tab_dataset is not None, 'If the dataset is not provided\
            to the function, a csv needs to have been provided when instatiating the class'
            dataset = self.tab_dataset
        if cat is None:
            con, cat = self.get_con_cat_columns(dataset)
        cat_ids = []
        for index, key in enumerate(dataset):
            if key in set(cat):
                cat_ids.append((index,
                                len(pd.unique(dataset[key])),
                                pd.unique(dataset[key])))
        return cat_ids

    def update_policy(self, trajectories):
        # Update the policy network using PPO
        if not trajectories:
            return 0, 0  # Return zeros if no trajectories

        try:
            states, actions, rewards = zip(*trajectories)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            
            # Check for NaN values
            if torch.isnan(states).any():
                states = torch.nan_to_num(states, nan=0.0)
            if torch.isnan(rewards).any():
                rewards = torch.nan_to_num(rewards, nan=0.0)
            
            # Normalize rewards for more stable learning
            if len(rewards) > 1 and rewards.std() > 0:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Multiple optimization epochs for each batch of data
            policy_losses = []
            value_losses = []
            
            for _ in range(5):  # Number of optimization epochs
                # Calculate advantages
                with torch.no_grad():
                    values = self.value_network(states).squeeze()
                    advantages = rewards - values
                    # Normalize advantages for more stable learning
                    if len(advantages) > 1 and advantages.std() > 0:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Get current action probabilities (before update)
                action_probs = self.policy_network(states)
                
                # Fix: Remove the extra dimension if it exists
                if action_probs.dim() > 2:
                    action_probs = action_probs.squeeze(1)
                
                # Check for NaN values
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    print("Warning: NaN or Inf values detected in action probabilities during update. Skipping update.")
                    return 0, 0
                
                # Ensure probabilities are valid
                action_probs = torch.clamp(action_probs, min=1e-6)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                
                # Get log probabilities of selected actions
                old_action_probs = action_probs.gather(1, actions).squeeze().detach()
                old_log_probs = torch.log(old_action_probs + 1e-10)  # Add small epsilon for numerical stability
                
                # PPO mini-batch update
                self.optimizer.zero_grad()
                
                # Get updated action probabilities
                new_action_probs = self.policy_network(states)
                if new_action_probs.dim() > 2:
                    new_action_probs = new_action_probs.squeeze(1)
                
                # Handle any NaN values
                new_action_probs = torch.clamp(new_action_probs, min=1e-6)
                new_action_probs = new_action_probs / new_action_probs.sum(dim=-1, keepdim=True)
                
                # Add entropy bonus to encourage exploration
                entropy = -torch.sum(new_action_probs * torch.log(new_action_probs + 1e-10), dim=1).mean()
                
                # Get log probabilities of selected actions with updated policy
                new_selected_probs = new_action_probs.gather(1, actions).squeeze()
                new_log_probs = torch.log(new_selected_probs + 1e-10)
                
                # Calculate the probability ratio and clip it
                ratio = torch.exp(new_log_probs - old_log_probs)
                # Clip to prevent extreme values
                ratio = torch.clamp(ratio, 0.1, 10.0)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages  # Clip ratio between 0.8 and 1.2
                new_values = self.value_network(states).squeeze()
                value_loss = nn.MSELoss()(new_values, rewards)
                
                # Combine losses with entropy bonus to encourage exploration
                entropy_coef = 0.01  # Entropy coefficient
                policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
                
                # Combine and backpropagate
                total_loss = policy_loss + 0.5 * value_loss
                
                # Check if loss is valid
                if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                    total_loss.backward()
                    
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
                    
                    self.optimizer.step()
                    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                else:
                    print("Warning: Invalid loss detected. Skipping backward pass.")
            
            if policy_losses:
                return np.mean(policy_losses), np.mean(value_losses)
            else:
                return 0, 0
                
        except Exception as e:
            print(f"Error in update_policy: {e}")
            return 0, 0

    def evaluate_policy(self, model, num_samples=10):
        """Evaluate the current policy by trying to find counterfactuals for a few samples."""
        if not hasattr(self, 'evaluation_success_rates'):
            self.evaluation_success_rates = []
        
        original_features = self.tab_dataset.values
        categories = self.generate_cats_ids(self.tab_dataset)
        
        # Sample a few instances for evaluation
        indices = np.random.choice(len(original_features), num_samples, replace=False)
        success_count = 0
        
        for i in indices:
            original = original_features[i]
            state = self.get_state(original, original.copy())
            found_cf = False
            
            # Try to find a counterfactual with limited steps
            for step in range(20):  # Limited steps for evaluation
                action = self.select_action(state)
                modified_features = self.apply_action(state, action, categories)
                next_state = self.get_state(original, modified_features)
                
                # Check if we found a counterfactual
                if self.generate_prediction(model, modified_features) != self.generate_prediction(model, original):
                    found_cf = True
                    success_count += 1
                    break
                
                state = next_state
        
        success_rate = success_count / num_samples
        self.evaluation_success_rates.append(success_rate)
        print(f"Generation {self.generation}: Evaluation success rate: {success_rate:.2f}")
        
        return success_rate

    def train(self, model, max_generations=20, samples_per_batch=32):
        """Train the PPO policy without focusing on finding final counterfactuals."""
        # Initialize training parameters
        self.max_generations = max_generations
        original_features = self.tab_dataset.values
        categories = self.generate_cats_ids(self.tab_dataset)
        
        # Initialize history tracking
        if not hasattr(self, 'policy_loss_history'):
            self.policy_loss_history = []
        if not hasattr(self, 'value_loss_history'):
            self.value_loss_history = []
        if not hasattr(self, 'evaluation_success_rates'):
            self.evaluation_success_rates = []
        
        for generation in tqdm(range(max_generations)):
            self.generation = generation
            all_trajectories = []
            
            # Sample instances for this training batch
            indices = np.random.choice(len(original_features), min(samples_per_batch, len(original_features)), replace=False)
            
            for i in indices:
                original = original_features[i]
                # Always start from the original instance during training
                state = self.get_state(original, original.copy())
                
                # Collect episode trajectories
                episode_states, episode_actions, episode_rewards = [], [], []
                
                for step in range(20):  # Steps per episode
                    try:
                        action = self.select_action(state)
                        modified_features = self.apply_action(state, action, categories)
                        next_state = self.get_state(original, modified_features)
                        reward = self.calculate_reward(original, modified_features, model, i)
                        
                        episode_states.append(state)
                        episode_actions.append(action)
                        episode_rewards.append(reward)
                        
                        state = next_state
                    except Exception as e:
                        print(f"Error in training step: {e}")
                        continue
                
                # Calculate returns and add to trajectories
                returns = []
                discounted_sum = 0
                gamma = 0.99
                for r in reversed(episode_rewards):
                    discounted_sum = r + gamma * discounted_sum
                    returns.insert(0, discounted_sum)
                
                for s, a, r in zip(episode_states, episode_actions, returns):
                    all_trajectories.append((s, a, r))
            
            # Update policy with all trajectories collected in this generation
            if all_trajectories:
                policy_loss, value_loss = self.update_policy(all_trajectories)
                self.policy_loss_history.append(policy_loss)
                self.value_loss_history.append(value_loss)
            
            # Evaluate policy performance periodically
            if generation % 2 == 0:
                try:
                    self.evaluate_policy(model)
                except Exception as e:
                    print(f"Error in evaluation: {e}")
        
        # Save the trained model
        try:
            self.save("models")
        except Exception as e:
            print(f"Error saving model: {e}")
            # Try to save to current directory if models directory doesn't exist
            try:
                self.save("./")
            except:
                print("Failed to save model")

    def generate_counterfactuals(self, model, max_steps=50):
        """Generate counterfactuals using the trained policy."""
        original_features = self.tab_dataset.values
        categories = self.generate_cats_ids(self.tab_dataset)
        self.results = [None] * len(original_features)
        self.best_distances = [float('inf')] * len(original_features)
        cfes = self.tab_dataset.copy().values
        
        for i, original in tqdm(enumerate(original_features), total=len(original_features), desc="Generating counterfactuals"):
            # Start from the original instance
            state = self.get_state(original, original.copy())
            best_cf = None
            best_distance = float('inf')
            
            # Try multiple episodes to find the best counterfactual
            for episode in range(3):  # Try a few episodes per instance
                current_state = state.copy()
                current_features = original.copy()
                
                for step in range(max_steps):
                    try:
                        action = self.select_action(current_state)
                        modified_features = self.apply_action(current_state, action, categories)
                        next_state = self.get_state(original, modified_features)
                        
                        # Check if we found a counterfactual
                        prediction_changed = self.generate_prediction(model, modified_features) != self.generate_prediction(model, original)
                        
                        if prediction_changed:
                            distance = self.get_distance(original, modified_features)
                            if distance < best_distance:
                                best_distance = distance
                                best_cf = modified_features
                        
                        current_state = next_state
                        current_features = modified_features
                    except Exception as e:
                        print(f"Error in generation step: {e}")
                        continue
                
            # Store the best counterfactual found
            if best_cf is not None:
                try:
                    decoded_features = self.decode_features(best_cf)
                    self.results[i] = (original, decoded_features, True, round(best_distance, 2), 0)
                    cfes[i] = decoded_features
                    self.best_distances[i] = best_distance
                except Exception as e:
                    print(f"Error updating results: {e}")
        
        # Display results
        self.display_results()
        con, cat = self.get_con_cat_columns(self.tab_dataset)
        self.display_KPIs(self.tab_dataset, cfes, con, cat, model)
        self.plot_summary()
        
        #self.calculate_feature_importance()


    def run_complete_workflow(self, model, training_generations=20, cf_max_steps=50):
        """Complete workflow: train policy first, then generate counterfactuals."""
        # Step 1: Train the policy
        print("Training PPO policy...")
        self.train(model, max_generations=training_generations)
        
        # Step 2: Generate counterfactuals using the trained policy
        print("Generating counterfactuals...")
        self.generate_counterfactuals(model, max_steps=cf_max_steps)
        
        # Plot training metrics
        self.plot_training_metrics()


    def run_inference_only(self, model, model_path="models/", max_steps=50):
        """
        Use a pre-trained PPO model to generate counterfactuals without additional training.
        
        Parameters:
        - model: The classifier model for which we're generating counterfactuals
        - model_path: Path to the pre-trained PPO model
        - max_steps: Maximum steps to take when generating each counterfactual
        """
        # Load the pre-trained model
        if not self.load(model_path):
            print("Failed to load pre-trained model. Please ensure the model exists at the specified path.")
            return False
        
        # Generate counterfactuals using the loaded model
        print("Generating counterfactuals using pre-trained policy...")
        self.generate_counterfactuals(model, max_steps=max_steps)
        
        return True



    def display_KPIs(self, x, y, con, cat, model):
        # Display KPIs for the generated counterfactuals
        proximity = proximity_KPI(x, y, con, cat)
        sparsity = sparsity_KPI(x, y)
        #validity = validity_KPI(model, x, y)

        print(f"Proximity KPI: {proximity:.4f}")
        print(f"Sparsity KPI: {sparsity:.4f}")
        #print(f"Validity KPI: {validity:.4f}")

    def plot_training_metrics(self):
        """Plot training metrics including policy loss, value loss, and evaluation success rates."""
        plt.figure(figsize=(15, 5))
        
        # Plot policy loss
        plt.subplot(1, 3, 1)
        plt.plot(self.policy_loss_history, label='Policy Loss', color='blue')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Policy Loss')
        plt.legend()
        
        # Plot value loss
        plt.subplot(1, 3, 2)
        plt.plot(self.value_loss_history, label='Value Loss', color='orange')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Value Loss')
        plt.legend()
        
        # Plot evaluation success rates
        if hasattr(self, 'evaluation_success_rates') and len(self.evaluation_success_rates) > 0:
            plt.subplot(1, 3, 3)
            plt.plot(range(0, len(self.evaluation_success_rates) * 5, 5), 
                    self.evaluation_success_rates, 
                    label='Success Rate', 
                    color='green')
            plt.xlabel('Generation')
            plt.ylabel('Success Rate')
            plt.title('Evaluation Success Rate')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_metrics(self):
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.policy_loss_history, label='Policy Loss', color='orange')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Policy Loss per Generation')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.value_loss_history, label='Value Loss', color='green')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Value Loss per Generation')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_summary(self):
        # Plot the summary of the results (distance to original)
        distances = [result[3] for result in self.results if result is not None]
        if distances:
            plt.figure(figsize=(8, 4))
            plt.hist(distances, bins=20, color='blue', alpha=0.7)
            plt.xlabel('Distance to Original')
            plt.ylabel('Frequency')
            plt.title('Distribution of Distances to Original')
            plt.show()

            mean_distance = np.mean(distances)
            median_distance = np.median(distances)
            # Plot the distance to original for each sample
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(self.results)), [result[3] if result is not None else float('inf') for result in self.results], color='blue', alpha=0.7)
            plt.xlabel('Sample Index')
            plt.ylabel('Distance to Original')
            plt.title('Distance to Original for Each Sample')
            plt.axhline(y=mean_distance, color='r', linestyle='--', label='Mean Distance')
            plt.axhline(y=median_distance, color='g', linestyle='--', label='Median Distance')
            plt.legend()
            plt.show()
        else:
            print("No successful counterfactuals found to plot.")

    def display_results(self):
        # Display the results in a readable format
        successful_count = 0
        for i, result in enumerate(self.results):
            if result is not None:
                successful_count += 1
        
        print(f"Found {successful_count} counterfactuals out of {len(self.results)} samples.")
        
        # Display top 5 counterfactuals with the smallest distances
        sorted_results = sorted([r for r in self.results if r is not None], key=lambda x: x[3] if x is not None else float('inf'))
        
        if sorted_results:
            print("\nTop 5 Counterfactuals with Smallest Distances:")
            for i, result in enumerate(sorted_results[:min(5, len(sorted_results))]):
                if result is not None:
                    original, modified, prediction_changed, distance, generation = result
                    print(f"Sample #{i+1}:")
                    print(f"  Original: {original}")
                    print(f"  Modified: {modified}")
                    print(f"  Distance: {distance}")
                    print(f"  Found in generation: {generation}")
            
            distances = [result[3] for result in self.results if result is not None]
            print("\nDistance metrics:")
            print(f"Mean distance: {np.mean(distances):.4f}")
            print(f"Median distance: {np.median(distances):.4f}")
            print(f"Max distance: {np.max(distances):.4f}")
            print(f"Min distance: {np.min(distances):.4f}\n")
        else:
            print("No successful counterfactuals found.")


    # Feature importance calculation based on the number of times each action was taken
    def calculate_feature_importance(self):
        feature_importance = {}
        features = self.tab_dataset.columns.tolist()
        
        for action, count in enumerate(self.actions_stats):
            if count > 0:
                feature_name = self.action_space[action]
                # Extract the feature name from the action name
                feature_name = '_'.join(feature_name.split('_')[1:])
                
                # Count the number of times this feature was modified
                if feature_name in features:
                    if feature_name not in feature_importance:
                        feature_importance[feature_name] = 0
                    feature_importance[feature_name] += count
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("Feature Importance based on Action Counts:")
        for feature, count in sorted_importance:
            print(f"{feature}: {count} times")

        # plot a bar chart of the feature importance
        plt.figure(figsize=(10, 5))
        plt.bar([f for f, _ in sorted_importance], [c for _, c in sorted_importance], color='blue', alpha=0.7)
        plt.xlabel("Feature")
        plt.ylabel("Importance (Action Count)")
        plt.title("Feature Importance based on Action Counts")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def save(self, path):
        '''Function to save the model and the dataset'''
        try:
            torch.save(self.policy_network.state_dict(), path + '/policy_network.pth')
            torch.save(self.value_network.state_dict(), path + '/value_network.pth')
            self.tab_dataset.to_csv(path + '/dataset.csv', index=False)
            print(f"Model and dataset saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, path):
        '''Function to load the pre-trained model'''
        try:
            self.policy_network.load_state_dict(torch.load(path + '/policy_network.pth'))
            self.value_network.load_state_dict(torch.load(path + '/value_network.pth'))
            self.tab_dataset = pd.read_csv(path + '/dataset.csv')
            print(f"Model loaded successfully from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    @classmethod
    def from_csv(cls, path):
        return cls(dataset_path=path)
    
    def transform_x_2_input(self, x, pytorch = True):
        '''Function to transform the raw input in the form of a pandas dataset
        or of a numpy array to the required format as input of the neural net(s)'''
        if isinstance(x, pd.DataFrame):
            x = x.copy()
            con, cat = self.get_con_cat_columns(x)
            if len(cat)>0:
                for feature in cat:
                    enc = LabelEncoder()
                    x[feature] = enc.fit(x[feature]).transform(x[feature])
            model_input = torch.tensor(x.values, dtype=torch.float) if pytorch else x.values
        elif isinstance(x, np.ndarray):
            model_input = torch.tensor(x, dtype = torch.float) if pytorch else x
        else:
            raise ValueError("The input x must be a pandas dataframe or a numpy array")
        return model_input
    
    def get_con_cat_columns(self, x):
        
        assert isinstance(x, pd.DataFrame), 'This method can be used only if input\
            is an instance of pandas dataframe at the moment.'
        
        con = []
        cat = []
        
        for column in x:
            if x[column].dtype == 'O':
                cat.append(column)
            else:
                con.append(column)
                
        return con, cat
