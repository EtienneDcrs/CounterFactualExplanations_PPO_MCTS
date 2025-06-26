import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import time
from stable_baselines3.common.callbacks import BaseCallback

class PPOEnv(gym.Env):
    def __init__(self, dataset_path=None, numpy_dataset=None, model=None, distance_metric=None):
        """
        Initialize the PPO environment for counterfactual generation.
        
        Parameters:
        - dataset_path: Path to the CSV dataset (optional)
        - numpy_dataset: Dataset as numpy array (optional)
        - model: The classification model we're generating counterfactuals for
        - distance_metric: Distance function to use (default: custom hybrid distance)
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
        
        # Get feature categories
        self.con_columns, self.cat_columns = self.get_con_cat_columns(self.tab_dataset)
        self.cats_ids = self.generate_cats_ids(self.tab_dataset)
        
        # KEY FIX 1: Create categorical mapping for proper encoding
        self.cat_mappings = self._create_categorical_mappings()
        
        # Precompute feature ranges for normalization (for continuous features only)
        self.feature_ranges = self._compute_feature_ranges()
        
        # Set distance metric
        if distance_metric is None:
            # Default to custom hybrid distance
            self.distance = lambda x, y: np.linalg.norm(x - y, ord=2)
            self.distance = self.calculate_distance
        else:
            self.distance = distance_metric
        
        # Track current state
        self.current_instance_idx = None
        self.original_features = None
        self.modified_features = None
        self.original_encoded = None
        self.modified_encoded = None
        self.sample_distance = None
        self.steps_taken = 0
        
        # Default constraints (which features can be modified)
        self.constraints = [1] * (len(self.tab_dataset.columns) - 1)  # Exclude target variable
        #self.constraints[10] = 0  # Example: feature at index 10 is fixed
        
        # Define action space based on features that can be modified
        self.define_action_space()
        
        # Store the action names separately before converting to Discrete space
        self.action_names = self.action_space.copy()
        
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
        self.max_steps = 500
        self.reward_range = (-100, 100)
        self.done = False

    # KEY FIX 1: Create proper categorical mappings for encoding
    def _create_categorical_mappings(self):
        """Create proper categorical mappings for encoding."""
        cat_mappings = {}
        
        for idx, n_cats, cat_values in self.cats_ids:
            column_name = self.tab_dataset.columns[idx]
            # Create mapping from category value to index
            cat_mappings[column_name] = {cat_val: i for i, cat_val in enumerate(cat_values)}
        
        return cat_mappings

    def _compute_feature_ranges(self):
        """Precompute feature ranges for efficient distance calculation."""
        feature_ranges = {}
        
        for i, column in enumerate(self.tab_dataset.columns[:-1]):  # Exclude target
            if column in self.con_columns:
                # For continuous features, store min and max values
                min_val = self.tab_dataset[column].min()
                max_val = self.tab_dataset[column].max()
                avg_val = self.tab_dataset[column].mean()
                feature_ranges[i] = {
                    'type': 'continuous',
                    'min': min_val,
                    'max': max_val,
                    'range': min(avg_val, max_val - min_val) 
                }
            else:
                # For categorical features, just mark as categorical
                feature_ranges[i] = {'type': 'categorical'}
        
        return feature_ranges
    
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

        self.original_encoded = self.encode_features(self.original_features)
        self.modified_encoded = self.encode_features(self.modified_features)

        # Get the original prediction and store it for later use
        self.original_prediction = self.generate_prediction(self.model, self.original_features)
        
        # Reset tracking variables
        self.steps_taken = 0
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
        
        # Calculate distance between original and modified features (using raw features)
        distance = self.calculate_distance(self.original_features, self.modified_features)
        
        if self.steps_taken == self.max_steps - 1 :
            print(f"Original features: {self.original_features}")
            print(f"Modified features: {self.modified_features}")
            print(f"Distance: {distance}")
    
        # Determine if we found a counterfactual (class changed)
        counterfactual_found = (modified_prediction != self.original_prediction)
        
        # Check if episode is done
        done = counterfactual_found or (self.steps_taken >= self.max_steps)
        
        # Calculate reward
        reward = self.calculate_reward(distance, counterfactual_found)
        
        # Get the next observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'distance': distance,
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
        # Add metadata to the observation: steps taken, distance, normalized progress
        distance = self.calculate_distance(self.original_features, self.modified_features)
        steps_normalized = self.steps_taken / self.max_steps
        
        # Store current distance for use in observation
        self.sample_distance = distance
        
        # Combine all into observation vector
        observation = np.concatenate([
            self.original_encoded,
            self.modified_encoded,
            [steps_normalized, self.sample_distance, float(self.original_prediction)]
        ]).astype(np.float32)
        
        # Replace any NaN values with 0
        observation = np.nan_to_num(observation, nan=0.0)
        
        return observation

    def calculate_distance(self, original_features, modified_features):
        """Calculate the distance between original and modified features using raw features."""
        """
        Calculate hybrid distance that treats categorical and continuous features differently.
        
        - Categorical features: 100% change if different, 0% if same
        - Continuous features: Relative percentage change normalized by feature range
        
        Args:
            original_features: Original feature values (raw, not encoded)
            modified_features: Modified feature values (raw, not encoded)
        
        Returns:
            float: Weighted distance score
        """
        if len(original_features) != len(modified_features):
            raise ValueError("Feature vectors must have the same length")
        
        total_distance = 0.0
        num_features = len(original_features)
        
        for i in range(num_features):
            feature_info = self.feature_ranges.get(i, {})
            orig_val = original_features[i]
            mod_val = modified_features[i]
            
            if feature_info['type'] == 'categorical' or isinstance(orig_val, str) or isinstance(orig_val, bool):
                # Categorical feature: 100% change if different, 0% if same
                if pd.isna(orig_val) and pd.isna(mod_val):
                # Both are NaN - no change
                    continue  # No contribution to distance
                elif pd.isna(orig_val) or pd.isna(mod_val):
                    # One is NaN, the other isn't - they differ
                    total_distance += 1.0
                elif orig_val != mod_val:
                    total_distance += 1.0  # 100% change
                
            else:  # continuous feature
                if pd.isna(orig_val) or pd.isna(mod_val):
                    total_distance += 1.0  # Consider NaN as a significant change
                # Continuous feature: relative percentage change
                elif orig_val != mod_val:
                    # Calculate relative change as percentage of feature range
                    try:
                        relative_change = max(mod_val, orig_val) / min(mod_val, orig_val) - 1 if min(mod_val, orig_val) != 0 else max(mod_val, orig_val)
                        total_distance += relative_change
                    except:
                        print(f"Error calculating relative change for feature {i}: {orig_val} -> {mod_val}")
                # else: feature has no variance, no contribution to distance
        
        # Normalize by number of features to get average distance per feature
        return total_distance

    # KEY FIX 5: Better reward structure
    def calculate_reward(self, distance, counterfactual_found, stage=1):
        """
        Calculate the reward for the current step based on the training stage.

        The reward function is structured to:
        1. Highly reward finding a valid counterfactual (Stage 1)
        2. Provide gradient through probability shifts (Stage 1)
        3. Penalize for distance from original (Stage 2)
        4. Penalize for excessive feature modifications (Stage 2)
        """

        # Stage 1: Finding Valid CFEs
        if counterfactual_found:
            # Higher reward for successful counterfactual
            base_reward = 100.0
            distance_reward = 10 / (distance + 1e-3)
            base_reward += distance_reward
            return max(base_reward, 10.0)  # Ensure minimum positive reward for success
        else:
            # If no counterfactual found, less harsh penalty to allow exploration
            reward = -0.5  # Reduced from -1
            return reward

    # KEY FIX 4: Improved action application with random categorical selection
    def apply_action(self, action_idx):
        """
        Apply the selected action to modify features, with improved categorical handling.

        Parameters:
            action_idx: Index of the action to take
            stage: Current training stage (1 or 2)

        Returns:
            modified_features: The features after applying the action
        """
        # Get the action name from the action names list
        action_name = self.action_names[action_idx]

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

                        # Get current category value (raw value, not encoded)
                        current_value = modified_features[feature_index]

                        # For categorical features, use random selection instead of deterministic cycling
                        available_values = [val for val in cat_values if val != current_value]

                        # Random selection for better exploration
                        if available_values:
                            next_value = np.random.choice(available_values)
                            modified_features[feature_index] = next_value

                # Handle continuous features with adaptive step sizes
                elif action_type in ['increase', 'decrease']:
                    current_value = modified_features[feature_index]
                    if current_value != 0:
                        base_step = abs(current_value) * 0.1  # 10% of current value
                    else:
                        # More reasonable step sizes based on feature range
                        feature_range = self.feature_ranges[feature_index]['range']
                        base_step = feature_range * 0.1  # 10% of feature range

                    # Apply the step
                    if action_type == 'increase':
                        max_value = self.tab_dataset.iloc[:, feature_index].max()
                        new_value = min(current_value + base_step, max_value)
                        modified_features[feature_index] = new_value
                    else:  # decrease
                        min_value = self.tab_dataset.iloc[:, feature_index].min()
                        new_value = max(current_value - base_step, min_value)
                        modified_features[feature_index] = new_value

                    # Round if the feature is integer type
                    if self.tab_dataset.iloc[:, feature_index].dtype in [np.int64, np.int32, int]:
                        modified_features[feature_index] = int(round(modified_features[feature_index]))

        except Exception as e:
            print(f"Error applying action {action_name}: {e}")
            # Return the unmodified features if there's an error

        return modified_features

    def define_action_space(self, constraints=None):
        # Define the action space based on the features
        self.action_space = []
        if constraints is None:
            constraints = self.constraints  # Use default constraints if not provided
        if constraints is None:
            constraints = [1] * len(self.tab_dataset.columns)  # Default to no constraints

        for i, column in enumerate(self.tab_dataset.columns[:-1]):  # Exclude the target variable
            # if the feature is constrained, skip it : fixed features aren't present in the action space
            if constraints[i] == 0:
                    continue
            if self.tab_dataset[column].dtype == 'O':  # Categorical feature
                self.action_space.append(f'change_{column}')
            else:  # Continuous feature
                self.action_space.append(f'increase_{column}')
                self.action_space.append(f'decrease_{column}')
        print(f"Action space defined : {self.action_space}")
        return self.action_space

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
        
    def encode_features(self, features):
        """Encode features for the model input with proper categorical mapping."""
        # Create a DataFrame for easier handling
        features_df = pd.DataFrame([features], columns=self.tab_dataset.columns[:-1])
        
        # Create a copy to avoid modifying the original
        encoded_features = features_df.copy()
        
        # FIXED: Proper categorical encoding using mappings
        for column_name, mapping in self.cat_mappings.items():
            if column_name in encoded_features.columns:
                # Get the current value
                current_val = encoded_features[column_name].iloc[0]
                
                # Map to index, default to 0 if not found
                encoded_val = mapping.get(current_val, 0)
                encoded_features[column_name] = encoded_val
        
        # Convert to numeric, handle NaNs
        encoded = encoded_features.astype(float).values.flatten()
        encoded = np.nan_to_num(encoded, nan=0.0)
        
        return encoded

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


class PPOMonitorCallback(BaseCallback):
    """
    Custom callback for monitoring PPO training progress.
    """
    def __init__(self, eval_env=None, eval_freq=1000, n_eval_episodes=5, verbose=1):
        super(PPOMonitorCallback, self).__init__(verbose)
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