import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class PPOEnv(gym.Env):
    def __init__(self, dataset_path=None, numpy_dataset=None, model=None, distance_metric=None):
        """
        Initialize the CERTIFAI environment for counterfactual generation.
        
        Parameters:
        - dataset_path: Path to the CSV dataset (optional)
        - numpy_dataset: Dataset as numpy array (optional)
        - model: The classification model we're generating counterfactuals for
        - distance_metric: Distance function to use (default: Euclidean distance)
        """
        super(PPOEnv, self).__init__()
        
        # Load dataset
        self.dataset_path = dataset_path
        self.numpy_dataset = numpy_dataset
        self.tab_dataset = None
        
        if dataset_path is not None:
            self.tab_dataset = pd.read_csv(dataset_path)
        elif numpy_dataset is not None:
            self.tab_dataset = pd.DataFrame(numpy_dataset)
        
        assert self.tab_dataset is not None, "Dataset must be provided through dataset_path or numpy_dataset"
        
        # Store the model
        self.model = model
        assert model is not None, "Classification model must be provided"
        
        # Set distance metric
        if distance_metric is None:
            # Default to Euclidean distance
            self.distance = lambda x, y: np.linalg.norm(x - y, ord=2)
        else:
            self.distance = distance_metric
        
        # Get feature categories
        self.con_columns, self.cat_columns = self.get_con_cat_columns(self.tab_dataset)
        self.cats_ids = self.generate_cats_ids(self.tab_dataset)
        
        # Track current state
        self.current_instance_idx = None
        self.original_features = None
        self.modified_features = None
        self.best_distance = float('inf')
        self.steps_taken = 0
        
        # Default constraints (which features can be modified)
        self.constraints = [1] * (len(self.tab_dataset.columns) - 1)  # Exclude target variable
        
        # For each categorical feature, set constraints to 0
        for feature,i in enumerate(self.tab_dataset.columns[:-1]):  # Exclude target variable
            if feature in self.cat_columns:
                self.constraints[i] = 0          

        # Define action space based on features that can be modified
        self.action_space = self.define_action_space()
        
        # Convert the list of action names to a proper gym action space
        self.action_space = spaces.Discrete(len(self.action_space))
        
        # Define observation space
        # Original features + modified features + metadata (steps, distance, etc.)
        feature_dim = len(self.tab_dataset.columns) - 1  # Exclude target variable
        obs_dim = feature_dim * 2 + 3  # Original + modified + metadata
        
        # Set observation space bounds based on feature ranges
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Episode settings
        self.max_steps = 50
        self.reward_range = (-100, 100)
        self.done = False

    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            observation: Initial state observation
        """
        # Select a random instance from the dataset
        self.current_instance_idx = np.random.randint(0, len(self.tab_dataset))
        
        # Get the original features (excluding target variable) and create a copy for modification
        self.original_features = self.tab_dataset.iloc[self.current_instance_idx].values[:-1]
        self.modified_features = self.original_features.copy()
        
        # Get the original prediction
        self.original_prediction = self.generate_prediction(self.model, self.original_features)
        
        # Reset tracking variables
        self.steps_taken = 0
        self.best_distance = float('inf')
        self.done = False
        
        # Create and return the initial observation
        return self._get_observation()

    def step(self, action):
        """
        Execute one step in the environment.
        
        Parameters:
            action: Index of the action to take
        
        Returns:
            observation: Next state observation
            reward: Reward for the action
            done: Whether the episode has ended
            info: Additional information for debugging
        """
        # Increment step counter
        self.steps_taken += 1
        
        # Apply the selected action to modify features
        self.modified_features = self.apply_action(action)
        
        # Get the new prediction
        modified_prediction = self.generate_prediction(self.model, self.modified_features)
        
        # Calculate distance between original and modified features
        distance = self.calculate_distance()
        
        # Update best distance if improved
        if distance < self.best_distance:
            self.best_distance = distance
        
        # Determine if we found a counterfactual (class changed)
        counterfactual_found = (modified_prediction != self.original_prediction)
        
        # Check if episode is done
        done = counterfactual_found or (self.steps_taken >= self.max_steps)
        
        # Calculate reward
        reward = self.calculate_reward(distance, counterfactual_found, modified_prediction)
        
        # Get the next observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'distance': distance,
            'best_distance': self.best_distance,
            'counterfactual_found': counterfactual_found,
            'original_prediction': self.original_prediction,
            'modified_prediction': modified_prediction,
            'steps_taken': self.steps_taken,
            'modified_features': self.modified_features
        }
        
        self.done = done
        return observation, reward, done, info

    def _get_observation(self):
        """Create observation vector from current state."""
        # Encode features for the observation
        original_encoded = self.encode_features(self.original_features)
        modified_encoded = self.encode_features(self.modified_features)
        
        # Add metadata to the observation: steps taken, distance, normalized progress
        distance = self.calculate_distance()
        steps_normalized = self.steps_taken / self.max_steps
        distance_normalized = min(distance / 10.0, 1.0)  # Normalize distance, cap at 1.0
        
        # Combine all into observation vector
        observation = np.concatenate([
            original_encoded, 
            modified_encoded,
            [steps_normalized, distance_normalized, float(self.original_prediction)]
        ]).astype(np.float32)
        
        # Replace any NaN values with 0
        observation = np.nan_to_num(observation, nan=0.0)
        
        return observation

    def calculate_distance(self):
        """Calculate the distance between original and modified features."""
        original_encoded = self.encode_features(self.original_features)
        modified_encoded = self.encode_features(self.modified_features)
        
        # Calculate distance using the specified distance metric
        try:
            distance = self.distance(original_encoded, modified_encoded)
            return distance
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return float('inf')

    def calculate_reward(self, distance, counterfactual_found, modified_prediction, stage=1):
        """
        Calculate the reward for the current step based on the training stage.

        The reward function is structured to:
        1. Highly reward finding a valid counterfactual (Stage 1)
        2. Provide gradient through probability shifts (Stage 1)
        3. Penalize for distance from original (Stage 2)
        4. Penalize for excessive feature modifications (Stage 2)
        """
        # Calculate the number of modified features (for sparsity)
        original_encoded = self.encode_features(self.original_features)
        modified_encoded = self.encode_features(self.modified_features)
        modified_features_count = np.sum(np.abs(original_encoded - modified_encoded) > 1e-6)

        # Stage 1: Finding Valid CFEs
        if stage == 1:
            if counterfactual_found:
                # Higher reward for successful counterfactual
                base_reward = 100.0
                distance_reward = 1000 / distance if distance > 0 else 0
                base_reward += distance_reward
                # Bonus for improving upon the best distance found so far
                if distance <= self.best_distance:
                    base_reward += 25.0
                return max(base_reward, 10.0)  # Ensure minimum positive reward for success
            else:
                # If no counterfactual found, penalize
                reward = -10
                return reward

        # Stage 2: Refining CFEs to Be Closer to the Original Sample
        elif stage == 2:
            if counterfactual_found:
                # Higher reward for successful counterfactual with smaller distance
                base_reward = 100.0
                # Reward based on distance
                distance_reward = 1000 / distance if distance > 0 else 0
                # Penalize for modifying too many features
                sparsity_penalty = min(modified_features_count * 2, 30)  # Cap at 30

                # Calculate final reward
                reward = base_reward + distance_reward - sparsity_penalty
                # Bonus for improving upon the best distance found so far
                if distance == self.best_distance:
                    reward += 25.0
                return max(reward, 10.0)  # Ensure minimum positive reward for success
            else:
                # If no counterfactual found, penalize
                reward = -10
                return reward

        else:
            raise ValueError("Invalid stage. Stage must be 1 or 2.")


    def apply_action(self, action_idx, stage=1):
        """
        Apply the selected action to modify features, with stage-specific behaviors.

        Parameters:
            action_idx: Index of the action to take
            stage: Current training stage (1 or 2)

        Returns:
            modified_features: The features after applying the action
        """
        # Get the action name from the action space
        action_name = self._action_idx_to_name(action_idx)

        # Create a copy of the current features to modify
        modified_features = self.modified_features.copy()

        try:
            # Extract feature name from action name
            if '_' in action_name:
                action_type = action_name.split('_')[0]  # 'change', 'increase', or 'decrease'
                feature_name = '_'.join(action_name.split('_')[1:])
                feature_index = list(self.tab_dataset.columns[:-1]).index(feature_name)

                # Handle categorical features
                if action_type == 'change':
                    for idx, ncat, cat_values in self.cats_ids:
                        # Skip if not applicable
                        if idx != feature_index or ncat <= 1:
                            continue

                        # Get current category index (assuming encoded)
                        current_value = modified_features[feature_index]

                        # For categorical features, systematically try different categories
                        all_indices = list(range(ncat))
                        if current_value in all_indices:
                            all_indices.remove(int(current_value))

                        # Select next category in a deterministic way
                        if all_indices:
                            # Choose next category based on step count for more systematic exploration
                            next_index = all_indices[self.steps_taken % len(all_indices)]
                            modified_features[feature_index] = next_index

                # Handle continuous features with adaptive step sizes
                elif action_type in ['increase', 'decrease']:
                    current_value = modified_features[feature_index]

                    # Adaptive step size based on progress through episode and training stage
                    progress = self.steps_taken / self.max_steps

                    # Stage-specific step size adjustments
                    if stage == 1:
                        # Stage 1: Larger steps for exploration
                        base_step = 0.20  # 20% change for broader exploration
                        fine_step = 0.10  # 10% change
                    else:
                        # Stage 2: Smaller steps for refinement
                        base_step = 0.10  # 10% change
                        fine_step = 0.03  # 3% change for fine-tuning

                    # Adaptive step size: larger at beginning, finer as we progress
                    step_size = base_step * (1 - progress) + fine_step * progress

                    # Apply the step
                    if action_type == 'increase':
                        max_value = self.tab_dataset.iloc[:, feature_index].max()
                        calculated_value = current_value * (1 + step_size)

                        # Round if the feature is integer type
                        if self.tab_dataset.iloc[:, feature_index].dtype in [np.int64, np.int32, int]:
                            calculated_value = int(calculated_value)

                        new_value = min(calculated_value, max_value)
                        modified_features[feature_index] = new_value
                    else:  # decrease
                        min_value = self.tab_dataset.iloc[:, feature_index].min()
                        calculated_value = current_value * (1 - step_size)

                        # Round if the feature is integer type
                        if self.tab_dataset.iloc[:, feature_index].dtype in [np.int64, np.int32, int]:
                            calculated_value = int(calculated_value)

                        new_value = max(calculated_value, min_value)
                        modified_features[feature_index] = new_value

        except Exception as e:
            print(f"Error applying action {action_name}: {e}")
            # Return the unmodified features if there's an error

        return modified_features

    def _action_idx_to_name(self, action_idx):
        """Convert action index to action name."""
        action_names = self.define_action_space()
        return action_names[action_idx]

    def encode_features(self, features):
        """Encode features for the model input."""
        # Create a DataFrame for easier handling
        features_df = pd.DataFrame([features], columns=self.tab_dataset.columns[:-1])
        
        # Create a copy to avoid modifying the original
        encoded_features = features_df.copy()
        
        # Encode categorical features
        for idx, n_cats, cat_values in self.cats_ids:
            if idx < len(encoded_features.columns):  # Ensure we don't go out of bounds
                for i, cat_value in enumerate(cat_values):
                    # Replace categorical values with their index
                    encoded_features.iloc[:, idx] = np.where(
                        encoded_features.iloc[:, idx] == cat_value, i, encoded_features.iloc[:, idx]
                    )
        
        # Convert to numeric, handle NaNs
        encoded = encoded_features.astype(float).values.flatten()
        encoded = np.nan_to_num(encoded, nan=0.0)
        
        return encoded

    def define_action_space(self, constraints=None):
        # Define the action space based on the features
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
                
        return action_space

    def generate_prediction(self, model, features):
        """Generate a prediction using the provided model."""
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

    def get_con_cat_columns(self, x):
        """Identify continuous and categorical columns in the dataset."""
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

    def generate_cats_ids(self, dataset = None, cat = None):
        """Generate categorical IDs for encoding."""
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


class CERTIFAIMonitorCallback(BaseCallback):
    """
    Custom callback for monitoring CERTIFAI training progress.
    """
    def __init__(self, eval_env=None, eval_freq=1000, n_eval_episodes=5, verbose=1):
        super(CERTIFAIMonitorCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.last_eval_time = time.time()
        self.counterfactuals_found = 0
        self.total_episodes = 0
        self.success_rate = 0.0

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Log time since last evaluation
            current_time = time.time()
            fps = self.eval_freq / (current_time - self.last_eval_time)
            self.last_eval_time = current_time
            
            if self.verbose > 1:
                print(f"Steps: {self.n_calls}, FPS: {fps:.2f}")
                print(f"Current success rate: {self.success_rate:.2%}")
            
            # Evaluate the model if an eval environment is provided
            # if self.eval_env is not None:
            #     self._evaluate_policy()
                
        return True
    
    def _evaluate_policy(self):
        """Evaluate the policy on the evaluation environment."""
        counterfactuals_found = 0
        
        for _ in range(self.n_eval_episodes):
            done = False
            obs = self.eval_env.reset()
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                
                if info['counterfactual_found']:
                    counterfactuals_found += 1
                    break
        
        self.total_episodes += self.n_eval_episodes
        self.counterfactuals_found += counterfactuals_found
        self.success_rate = self.counterfactuals_found / self.total_episodes
        
        if self.verbose > 0:
            print(f"Evaluation over {self.n_eval_episodes} episodes:")
            print(f"Success rate: {counterfactuals_found/self.n_eval_episodes:.2%}")
            print(f"Overall success rate: {self.success_rate:.2%}")