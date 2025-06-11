import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import time
from stable_baselines3.common.callbacks import BaseCallback

class GRPOEnv(gym.Env):
    def __init__(self, dataset_path=None, numpy_dataset=None, model=None):
        """
        Initialize the GRPO environment for counterfactual generation.
        Similar to PPOEnv but designed for group-based advantage computation.

        Parameters:
        - dataset_path: Path to the CSV dataset (optional)
        - numpy_dataset: Dataset as numpy array (optional)
        - model: The classification model we're generating counterfactuals for
        """
        super(GRPOEnv, self).__init__()

        # Load dataset
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

        # Track current state
        self.current_instance_idx = None
        self.original_features = None
        self.modified_features = None
        self.original_encoded = None
        self.modified_encoded = None
        self.sample_distance = None
        self.steps_taken = 0

        # Group sampling state for GRPO
        self.group_states = []
        self.group_rewards = []
        self.group_actions = []

        # Default constraints (which features can be modified)
        self.constraints = [1] * (len(self.tab_dataset.columns) - 1)  # Exclude target variable

        # For each categorical feature, set constraints to 0
        for i, features in enumerate(self.tab_dataset.columns[:-1]):  # Exclude target variable
            if self.tab_dataset[features].dtype == 'O':
                self.constraints[i] = 0

        # Define action space based on features that can be modified
        self.action_names = self.define_action_space()
        self.action_space = spaces.Discrete(len(self.action_names))

        # Define observation space
        feature_dim = len(self.tab_dataset.columns) - 1  # Exclude target variable
        obs_dim = feature_dim * 2 + 3  # Original + modified + metadata

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Episode settings
        self.max_steps = 100
        self.done = False

    def reset(self, instance_idx=None):
        """
        Reset the environment for a new episode.
        
        Parameters:
            instance_idx: Specific instance index to use (for group sampling)
        """
        # Select instance
        if instance_idx is not None:
            self.current_instance_idx = instance_idx
        else:
            self.current_instance_idx = np.random.randint(0, len(self.tab_dataset))

        # Get the original features and create a copy for modification
        self.original_features = self.tab_dataset.iloc[self.current_instance_idx].values[:-1]
        self.modified_features = self.original_features.copy()

        self.original_encoded = self.encode_features(self.original_features)
        self.modified_encoded = self.encode_features(self.modified_features)

        # Get the original prediction
        self.original_prediction = self.generate_prediction(self.model, self.original_features)

        # Reset tracking variables
        self.steps_taken = 0
        self.sample_distance = 0
        self.done = False

        # Clear group data for new episode
        self.group_states = []
        self.group_rewards = []
        self.group_actions = []

        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment."""
        self.steps_taken += 1
        
        # Apply the selected action to modify features
        self.modified_features = self.apply_action(action)
        self.modified_encoded = self.encode_features(self.modified_features)

        # Get the new prediction
        modified_prediction = self.generate_prediction(self.model, self.modified_features)

        # Calculate distance between original and modified features
        self.sample_distance = self.calculate_normalized_distance()
        distance = self.sample_distance

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
            'counterfactual_found': counterfactual_found,
            'original_prediction': self.original_prediction,
            'modified_prediction': modified_prediction,
            'steps_taken': self.steps_taken,
            'modified_features': self.modified_features,
            'reward': reward  # Store raw reward for group processing
        }

        self.done = done
        return observation, reward, done, info

    def sample_group_trajectories(self, policy_fn, group_size=4, max_steps=None):
        """
        Sample a group of trajectories for the same initial state.
        This is the core of GRPO - we need multiple outcomes to compare.
        
        Parameters:
            policy_fn: Function that takes observation and returns action
            group_size: Number of trajectories to sample
            max_steps: Maximum steps per trajectory
            
        Returns:
            List of trajectory dictionaries with rewards and actions
        """
        if max_steps is None:
            max_steps = self.max_steps
            
        trajectories = []
        initial_state = self.reset()  # Start from same initial state
        initial_idx = self.current_instance_idx
        
        for _ in range(group_size):
            # Reset to same initial state
            self.reset(instance_idx=initial_idx)
            
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'done': False,
                'total_reward': 0,
                'counterfactual_found': False
            }
            
            obs = self._get_observation()
            steps = 0
            
            while not self.done and steps < max_steps:
                # Get action and log probability from policy
                if hasattr(policy_fn, 'predict_with_log_prob'):
                    action, log_prob = policy_fn.predict_with_log_prob(obs)
                elif hasattr(policy_fn, 'predict'):
                    action, _ = policy_fn.predict(obs, deterministic=False)
                    log_prob = 0  # Will need to compute this separately
                else:
                    try:
                        # Try to get both action and log_prob
                        action, log_prob = policy_fn(obs)
                    except Exception as e:
                        # If it doesn't return log_prob or any other error
                        action = policy_fn(obs)
                        log_prob = 0
            
                trajectory['states'].append(obs.copy())
                trajectory['actions'].append(action)
                trajectory['log_probs'].append(log_prob)
                
                obs, reward, done, info = self.step(action)
                
                trajectory['rewards'].append(reward)
                trajectory['total_reward'] += reward
                
                if info['counterfactual_found']:
                    trajectory['counterfactual_found'] = True
                    trajectory['distance'] = info['distance']
                    trajectory['steps_to_success'] = steps + 1
                
                steps += 1
            
            trajectory['done'] = self.done
            trajectory['final_info'] = info if 'info' in locals() else {}
            trajectories.append(trajectory)
        
        return trajectories

    def compute_group_advantages(self, trajectories):
        """
        Compute advantages based on group comparison (core of GRPO).
        
        Parameters:
            trajectories: List of trajectory dictionaries
            
        Returns:
            List of advantages for each trajectory
        """
        # Extract total rewards from trajectories
        total_rewards = [traj['total_reward'] for traj in trajectories]
        
        if len(total_rewards) == 0:
            return []
        
        # Compute group statistics
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards) if len(total_rewards) > 1 else 1.0
        
        # Avoid division by zero
        if std_reward < 1e-8:
            std_reward = 1.0
        
        # Compute advantages: (reward - mean) / std
        advantages = [(reward - mean_reward) / std_reward for reward in total_rewards]
        
        return advantages

    def _get_observation(self):
        """Create observation vector from current state."""
        steps_normalized = self.steps_taken / self.max_steps

        observation = np.concatenate([
            self.original_encoded,
            self.modified_encoded,
            [steps_normalized, self.sample_distance, float(self.original_prediction)]
        ]).astype(np.float32)

        observation = np.nan_to_num(observation, nan=0.0)
        return observation

    def calculate_normalized_distance(self):
        """Calculate normalized distance between original and modified features."""
        normalized_distance = 1.0
        
        for orig, mod in zip(self.original_encoded, self.modified_encoded):
            if orig == mod:
                continue
            orig = abs(orig)
            mod = abs(mod)
            if orig == 0 or mod == 0:
                ratio = orig + mod
            else:
                ratio = orig / mod if orig > mod else mod / orig
                normalized_distance *= ratio

        return normalized_distance

    def calculate_reward(self, distance, counterfactual_found, modified_prediction):
        """Calculate a more nuanced reward for the current step."""
        if counterfactual_found:
            # Base reward for finding counterfactual
            base_reward = 100.0
            
            # Distance bonus - reward closer counterfactuals more
            distance_reward = max(10.0, 50.0 / max(distance, 1))
            
            # Step efficiency bonus - reward finding it faster
            step_bonus = max(0, 20 * (1 - self.steps_taken / self.max_steps))
            
            total_reward = base_reward + distance_reward + step_bonus
            return total_reward
        else:
            exploration_penalty = -1.0  # Instead of -10
            
            # Small reward for getting closer to decision boundary
            # This requires tracking prediction confidence, but helps with sparse rewards
            progress_reward = 0.0
            
            return exploration_penalty + progress_reward

    def apply_action(self, action_idx):
        """Apply the selected action to modify features."""
        action_name = self._action_idx_to_name(action_idx)
        modified_features = self.modified_features.copy()

        try:
            if '_' in action_name:
                action_type = action_name.split('_')[0]
                feature_name = '_'.join(action_name.split('_')[1:])
                feature_index = list(self.tab_dataset.columns[:-1]).index(feature_name)

                # Handle categorical features
                if action_type == 'change':
                    for idx, ncat, cat_values in self.cats_ids:
                        if idx != feature_index or ncat <= 1:
                            continue

                        current_value = modified_features[feature_index]
                        all_indices = list(range(ncat))
                        if current_value in all_indices:
                            all_indices.remove(int(current_value))

                        if all_indices:
                            next_index = all_indices[self.steps_taken % len(all_indices)]
                            modified_features[feature_index] = next_index

                # Handle continuous features
                elif action_type in ['increase', 'decrease']:
                    current_value = modified_features[feature_index]
                    progress = self.steps_taken / self.max_steps

                    base_step = 0.20
                    fine_step = 0.10
                    step_size = base_step * (1 - progress) + fine_step * progress

                    if action_type == 'increase':
                        max_value = self.tab_dataset.iloc[:, feature_index].max()
                        mean_value = self.tab_dataset.iloc[:, feature_index].mean()
                        calculated_value = current_value * (1 + step_size)
                        if calculated_value == current_value:
                            calculated_value = mean_value * step_size

                        if self.tab_dataset.iloc[:, feature_index].dtype in [np.int64, np.int32, int]:
                            calculated_value = int(calculated_value)
                            if calculated_value == current_value:
                                calculated_value += 1

                        new_value = min(calculated_value, max_value)
                        modified_features[feature_index] = new_value
                    else:  # decrease
                        min_value = self.tab_dataset.iloc[:, feature_index].min()
                        mean_value = self.tab_dataset.iloc[:, feature_index].mean()
                        calculated_value = current_value * (1 - step_size)
                        if calculated_value == current_value:
                            calculated_value = mean_value * step_size

                        if self.tab_dataset.iloc[:, feature_index].dtype in [np.int64, np.int32, int]:
                            calculated_value = int(calculated_value)
                            if calculated_value == current_value:
                                calculated_value -= 1

                        new_value = max(calculated_value, min_value)
                        modified_features[feature_index] = new_value
        except Exception as e:
            print(f"Error applying action {action_name}: {e}")

        return modified_features

    def _action_idx_to_name(self, action_idx):
        """Convert action index to action name."""
        return self.action_names[action_idx]

    def encode_features(self, features):
        """Encode features for the model input."""
        features_df = pd.DataFrame([features], columns=self.tab_dataset.columns[:-1])
        encoded_features = features_df.copy()

        for idx, n_cats, cat_values in self.cats_ids:
            if idx < len(encoded_features.columns):
                for i, cat_value in enumerate(cat_values):
                    encoded_features.iloc[:, idx] = np.where(
                        encoded_features.iloc[:, idx] == cat_value, i, encoded_features.iloc[:, idx]
                    )

        encoded = encoded_features.astype(float).values.flatten()
        encoded = np.nan_to_num(encoded, nan=0.0)
        return encoded

    def define_action_space(self, constraints=None):
        """Define the action space based on the features."""
        action_space = []
        if constraints is None:
            constraints = self.constraints

        for i, column in enumerate(self.tab_dataset.columns[:-1]):
            if constraints[i] == 0:
                continue
            if self.tab_dataset[column].dtype == 'O':
                action_space.append(f'change_{column}')
            else:
                action_space.append(f'increase_{column}')
                action_space.append(f'decrease_{column}')
        return action_space

    def generate_prediction(self, model, features):
        """Generate a prediction using the provided model."""
        try:
            features = self.encode_features(features)
            features = np.nan_to_num(features, nan=0.0)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model(features_tensor).argmax(dim=-1).item()
            return prediction
        except Exception as e:
            return 0

    def get_con_cat_columns(self, x):
        """Identify continuous and categorical columns in the dataset."""
        assert isinstance(x, pd.DataFrame), 'Input must be a pandas DataFrame'

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
            dataset = self.tab_dataset
        if cat is None:
            con, cat = self.get_con_cat_columns(dataset)
        
        cat_ids = []
        for index, key in enumerate(dataset):
            if key in set(cat):
                cat_ids.append((index, len(pd.unique(dataset[key])), pd.unique(dataset[key])))
        return cat_ids


class GRPOMonitorCallback(BaseCallback):
    """Custom callback for monitoring GRPO training progress."""
    
    def __init__(self, eval_env=None, eval_freq=1000, n_eval_episodes=5, verbose=1):
        super(GRPOMonitorCallback, self).__init__(verbose)
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