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
    def __init__(self, dataset_path=None, numpy_dataset=None, model=None, distance_metric=None, 
                 label_encoders=None, scaler=None, constraints=None, use_random_sampling=True):
        """
        Initialize the PPO environment for counterfactual generation.
        
        Parameters:
        - dataset_path: Path to the CSV dataset (optional)
        - numpy_dataset: Dataset as numpy array (optional)
        - model: The classification model we're generating counterfactuals for
        - distance_metric: Distance function to use (default: custom hybrid distance)
        - label_encoders: Label encoders for categorical features
        - scaler: Scaler for continuous features
        - constraints: Dictionary specifying feature constraints (e.g., {"age": "increase", "education_number": "fixed"})
        - use_random_sampling: Boolean flag to control whether reset uses random sampling (True) or fixed index (False)
        """
        super(PPOEnv, self).__init__()
        
        # Load dataset
        self.dataset_path = dataset_path
        self.numpy_dataset = numpy_dataset
        self.tab_dataset = None
        self.label_encoders = label_encoders
        self.scaler = scaler
        self.constraints = constraints or {}  # Store constraints dictionary, default to empty dict
        self.use_random_sampling = use_random_sampling  # New flag for sampling behavior
        
        if dataset_path is not None:
            self.tab_dataset = pd.read_csv(dataset_path)
            # Drop rows with NaN values
            original_len = len(self.tab_dataset)
            self.tab_dataset = self.tab_dataset.dropna()
            dropped_rows = original_len - len(self.tab_dataset)
            if dropped_rows > 0:
                print(f"Dropped {dropped_rows} rows with NaN values ({dropped_rows/original_len:.1%} of dataset)")
        elif numpy_dataset is not None:
            self.tab_dataset = pd.DataFrame(numpy_dataset)
            # Drop rows with NaN values
            original_len = len(self.tab_dataset)
            self.tab_dataset = self.tab_dataset.dropna()
            dropped_rows = original_len - len(self.tab_dataset)
            if dropped_rows > 0:
                print(f"Dropped {dropped_rows} rows with NaN values ({dropped_rows/original_len:.1%} of dataset)")
        
        assert self.tab_dataset is not None, "Dataset must be provided through dataset_path or numpy_dataset"
        self.feature_order = list(self.tab_dataset.columns[:-1])
        # Store the model
        self.model = model
        assert model is not None, "Classification model must be provided"
        
        # Get feature categories
        self.con_columns, self.cat_columns = self.get_con_cat_columns(self.tab_dataset)
        self.cats_ids = self.generate_cats_ids(self.tab_dataset)
        self.categorical_indices = [i for i, col in enumerate(self.tab_dataset.columns[:-1]) if col in self.cat_columns]
                
        # Set distance metric
        if distance_metric is None:
            # Default to custom hybrid distance
            self.distance = self.calculate_distance
        else:
            self.distance = distance_metric
        
        # Track current state
        self.current_instance_idx = None
        self.original_features = None
        self.modified_features = None
        self.original_encoded = None
        self.modified_encoded = None
        self.steps_taken = 0
        
        # Convert the list of action names to a proper gym action space
        self.action_names = self.define_action_space()
        self.action_space = spaces.Discrete(len(self.action_names))
        
        # Define observation space
        # Original features + modified features + metadata (steps, distance, etc.)
        feature_dim = len(self.tab_dataset.columns) - 1  # Exclude target variable
        obs_dim = feature_dim * 2 + 4  # Original + modified + metadata
        
        # Set observation space bounds based on feature ranges
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Episode settings
        self.max_steps = 100
        self.done = False

    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            observation: Initial state observation
        """
        # Select instance based on use_random_sampling flag
        if self.use_random_sampling:
            self.current_instance_idx = np.random.randint(0, len(self.tab_dataset))
        else:
            if self.current_instance_idx is None or self.current_instance_idx < 0 or self.current_instance_idx >= len(self.tab_dataset):
                raise ValueError(f"Invalid current_instance_idx: {self.current_instance_idx}. Must be set to a valid index when use_random_sampling=False")
        
        # Get the original features (excluding target variable) and create a copy for modification
        self.original_features = self.tab_dataset.iloc[self.current_instance_idx].values[:-1]
        self.modified_features = self.original_features.copy()

        self.original_encoded = self.encode_features(self.original_features)
        self.modified_encoded = self.encode_features(self.modified_features)

        # Get the original prediction and store it for later use
        self.original_prediction = self.generate_prediction(self.model, self.original_features)
        
        # Detect number of output classes (only once)
        if not hasattr(self, 'model_output_dim'):
            test_features = self.encode_features(self.original_features)
            with torch.no_grad():
                output = self.model(torch.tensor(test_features, dtype=torch.float32).unsqueeze(0))
            self.model_output_dim = output.shape[-1]

        # Update observation_space dynamically
        if not hasattr(self, '_obs_space_updated'):
            full_obs_dim = len(self._get_observation())
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(full_obs_dim,),
                dtype=np.float32
            )
            self._obs_space_updated = True

        # Reset tracking variables
        self.steps_taken = 0
        self.done = False
        
        # Create and return the initial observation
        return self._get_observation()

    def step(self, action):
        """
        Execute one step in the environment with dynamic action randomization based on steps taken.
        
        Parameters:
            action: Index of the action to take (may be overridden with random selection)
        
        Returns:
            observation: Next state observation
            reward: Reward for the action
            done: Whether the episode has ended
            info: Additional information for debugging
        """
        # Increment step counter
        self.steps_taken += 1
        randomisation_threshold = 40  # Threshold for randomization
        
        # Compute temperature for action randomization
        if self.steps_taken < randomisation_threshold:
            temperature = 0.05  # Low temperature: trust the policy
        else:
            # Linearly increase temperature from 0.1 to 10.0 between steps 80 and 100
            progress = (self.steps_taken - randomisation_threshold) / (self.max_steps - randomisation_threshold)
            temperature = 0.05 + (10.0 - 0.05) * progress

        # Get action probabilities from the PPO policy
        action_probs = self.get_action_probs(action, temperature)

        # Sample action from modified probabilities
        action = np.random.choice(self.action_space.n, p=action_probs)

        # Apply the selected action
        self.modified_features = self.apply_action(action)
        
        # Get the new prediction
        modified_prediction = self.generate_prediction(self.model, self.modified_features)
        
        # Calculate distance
        distance = self.calculate_distance(self.original_features, self.modified_features)
        
        # Check if counterfactual found
        counterfactual_found = (modified_prediction != self.original_prediction)
        
        # Check if episode is done
        done = counterfactual_found or (self.steps_taken >= self.max_steps)
        
        # Calculate reward
        reward = self.calculate_reward(distance, counterfactual_found)
        
        # Get next observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'distance': distance,
            'counterfactual_found': counterfactual_found,
            'original_prediction': self.original_prediction,
            'modified_prediction': modified_prediction,
            'steps_taken': self.steps_taken,
            'modified_features': self.modified_features,
            'action_taken': self._action_idx_to_name(action)
        }
        
        self.done = done
        return observation, reward, done, info

    def get_action_probs(self, action, temperature):
        """
        Compute action probabilities with temperature-based randomization.
        
        Parameters:
            action: The action suggested by the PPO policy
            temperature: Temperature parameter to control randomness
        
        Returns:
            probs: Modified action probabilities
        """
        # Get the PPO model's action logits (unnormalized probabilities)
        # Note: This requires access to the model's policy, which isn't directly exposed in stable_baselines3
        # As a workaround, we assume the provided action is the most likely and soften the distribution
        n_actions = self.action_space.n
        probs = np.ones(n_actions) / n_actions  # Start with uniform distribution
        probs[action] += 1.0  # Bias toward the suggested action

        # Apply temperature scaling
        probs = np.exp(np.log(probs + 1e-10) / temperature)
        probs = probs / (np.sum(probs) + 1e-10)  # Normalize to sum to 1
        return probs

    def _get_observation(self):
        """Create observation vector from current state."""
        # Distance L1 (already optimized)
        distance = self.calculate_distance(self.original_features, self.modified_features)
        steps_normalized = self.steps_taken / self.max_steps
        self.sample_distance = distance

        # One-hot encoding of the original class
        target_vector = np.zeros(self.model_output_dim, dtype=np.float32)
        target_vector[self.original_prediction] = 1.0

        # Final observation
        observation = np.concatenate([
            self.original_encoded,
            self.modified_encoded,
            [steps_normalized, self.sample_distance],
            target_vector
        ]).astype(np.float32)

        return np.nan_to_num(observation, nan=0.0)

    def calculate_distance(self, original_features, modified_features):
        """
        Calculate L1 (Manhattan) distance between encoded original and modified features.
        """
        dist = 0
        for i, (o, m) in enumerate(zip(original_features, modified_features)):
            if i in self.categorical_indices:
                dist += float(o != m)  # 1 if changed
            else:
                dist += abs(o - m)
        return dist

    def calculate_reward(self, distance, counterfactual_found):
        # Get model probabilities for modified features
        features_encoded = self.encode_features(self.modified_features)
        features_tensor = torch.tensor(features_encoded, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        
        # Reward based on reducing confidence in original class
        original_class_prob = probs[self.original_prediction]
        confidence_reward = 10 - 10 * original_class_prob  # Higher reward for lower confidence
        
        # Number of features modified
        num_modified_features = sum(1 for o, m in zip(self.original_features, self.modified_features) if o != m)
        # Reward based on number of features modified
        num_features_reward = 0 * num_modified_features if counterfactual_found else 0.0

        # Counterfactual bonus
        counterfactual_bonus = 100.0 if counterfactual_found else 0.0
        
        # Total reward
        reward = confidence_reward + num_features_reward + counterfactual_bonus
        return reward

    def apply_action(self, action_idx):
        """
        Apply the selected action to modify features, with stage-specific behaviors.

        Parameters:
            action_idx: Index of the action to take

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

                        # Get current category value (raw value, not encoded)
                        current_value = modified_features[feature_index]

                        # For categorical features, systematically try different categories
                        available_values = [val for val in cat_values if val != current_value]

                        # Select next category in a deterministic way
                        if available_values:
                            next_value = np.random.choice(available_values)
                            modified_features[feature_index] = next_value

                # Handle continuous features with adaptive step sizes
                elif action_type in ['increase', 'decrease']:
                    current_value = modified_features[feature_index]

                    # Adaptive step size based on progress through episode
                    progress = self.steps_taken / self.max_steps

                    # Adaptive step size: larger at beginning, finer as we progress
                    step_size = np.random.uniform(0.05, 0.25) * (1 - progress) + np.random.uniform(0.05, 0.15) * progress
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
        return self.action_names[action_idx]

    def encode_features(self, features):
        X = np.array(features, dtype=object).reshape(1, -1)  # shape (1, n)
        
        if self.label_encoders:
            for i, col in enumerate(self.feature_order):
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    val = str(X[0, i])
                    if val in encoder.classes_:
                        X[0, i] = encoder.transform([val])[0]
                    else:
                        X[0, i] = 0  # fallback

        if self.scaler:
            X = self.scaler.transform(X)

        return X.flatten().astype(np.float32)

    def define_action_space(self):
        """
        Define the action space based on the features and constraints.
        Features in constraints with 'fixed' are excluded, 'increase' or 'decrease' limit continuous features,
        and unconstrained features allow full modification.
        """
        action_space = []
        
        for column in self.tab_dataset.columns[:-1]:  # Exclude the target variable
            # Check if the feature has a constraint
            constraint = self.constraints.get(column, None)
            
            # If feature is fixed, skip it (no actions added)
            if constraint == "fixed":
                continue
                
            if self.tab_dataset[column].dtype == 'O':  # Categorical feature
                # Categorical features are only modified via 'change' action, unless fixed
                if constraint in [None, "change"]:  # Allow change if unconstrained or explicitly allowed
                    action_space.append(f'change_{column}')
            else:  # Continuous feature
                # Add actions based on constraint
                if constraint == "increase":
                    action_space.append(f'increase_{column}')
                elif constraint == "decrease":
                    action_space.append(f'decrease_{column}')
                elif constraint is None:  # No constraint, allow both increase and decrease
                    action_space.append(f'increase_{column}')
                    action_space.append(f'decrease_{column}')
        print(f"Defined action space with {len(action_space)} actions: {action_space}")
        return action_space

    def generate_prediction(self, model, features):
        try:
            features_encoded = self.encode_features(features)
            features_tensor = torch.tensor(features_encoded, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(features_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                pred_class = np.argmax(probs)
            return pred_class
        except Exception as e:
            print(f"Error in generate_prediction: {e}")
            return 0

    def get_con_cat_columns(self, x):
        """Identify continuous and categorical columns in the dataset."""
        assert isinstance(x, pd.DataFrame), 'This method can be used only if input is an instance of pandas dataframe at the moment.'
        
        con = []
        cat = []
        
        for column in x:
            if x[column].dtype == 'O':
                cat.append(column)
            else:
                con.append(column)
                
        return con, cat

    def generate_cats_ids(self, dataset=None, cat=None):
        """Generate categorical IDs for encoding."""
        if dataset is None:
            assert self.tab_dataset is not None, (
                'If the dataset is not provided to the function, '
                'a csv needs to have been provided when instantiating the class'
            )
            dataset = self.tab_dataset
        if cat is None:
            _, cat = self.get_con_cat_columns(dataset)

        cat_ids = []
        cat_maps = {}
        for index, key in enumerate(dataset):
            if key in set(cat):
                unique_vals = pd.unique(dataset[key])
                cat_ids.append((index, len(unique_vals), unique_vals))
                cat_maps[key] = {val: i for i, val in enumerate(unique_vals)}

        self.cat_maps = cat_maps  
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