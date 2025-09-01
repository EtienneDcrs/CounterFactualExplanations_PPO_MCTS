import time
import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import logging
from typing import Optional, Dict, List, Tuple, Callable
from stable_baselines3.common.callbacks import BaseCallback

logging.basicConfig(level=logging.INFO)

class DatasetHandler:
    def __init__(self, dataset_path: str, verbose: int = 1):
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.tab_dataset = self._load_dataset(dataset_path)
        self.feature_order = list(self.tab_dataset.columns[:-1])

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        dataset = pd.read_csv(dataset_path)
        original_len = len(dataset)
        dataset = dataset.dropna()
        if dropped_rows := original_len - len(dataset):
            self._log(f"Dropped {dropped_rows} rows with NaN values ({dropped_rows/original_len:.1%} of dataset)")
        return dataset

    def get_con_cat_columns(self) -> Tuple[List[str], List[str]]:
        feature_cols = list(self.tab_dataset.columns[:-1])
        continuous_columns = [col for col in feature_cols if self.tab_dataset[col].dtype != 'O']
        categorical_columns = [col for col in feature_cols if self.tab_dataset[col].dtype == 'O']
        return continuous_columns, categorical_columns

    def _log(self, message: str, level: str = 'info') -> None:
        if self.verbose >= (1 if level == 'info' else 2):
            getattr(self.logger, level)(message)

class GRPOEnv(gym.Env):
    RANDOMISATION_THRESHOLD: int = 40
    INITIAL_TEMPERATURE: float = 0.05
    MAX_TEMPERATURE: float = 10.0

    def __init__(self, dataset_path: Optional[str] = None, numpy_dataset: Optional[np.ndarray] = None,
                 model=None, distance_metric: Optional[Callable] = None, label_encoders: Optional[Dict] = None,
                 scaler=None, constraints: Optional[Dict[str, str]] = None, use_random_sampling: bool = True,
                 max_steps: int = 100,
                 verbose: int = 1):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

        if numpy_dataset is not None:
            self.tab_dataset = pd.DataFrame(numpy_dataset)
        else:
            self.dataset_handler = DatasetHandler(dataset_path, verbose)
            self.tab_dataset = self.dataset_handler.tab_dataset
        self.feature_order = list(self.tab_dataset.columns[:-1])

        self.model = model
        if label_encoders:
            self.label_encoders = {col: enc for col, enc in label_encoders.items() if col in self.feature_order}
        else:
            self.label_encoders = {}
        self.scaler = scaler
        self.constraints = constraints or {}
        self.use_random_sampling = use_random_sampling
        self.distance = distance_metric or self.calculate_distance

        self.continuous_columns, self.categorical_columns = self.get_con_cat_columns()
        self.categorical_indices = [self.feature_order.index(col) for col in self.categorical_columns]
        self.categorical_metadata = self.generate_cats_ids()

        assert model is not None, "Classification model must be provided"
        assert callable(getattr(model, '__call__', None)), "Model must be callable"

        self.action_names = self.define_action_space()
        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(len(self.feature_order) * 2 + 4,), dtype=np.float32)

        self.current_instance_idx = None
        self.original_features = None
        self.modified_features = None
        self.original_encoded = None
        self.modified_encoded = None
        self.steps_taken = 0
        self.max_steps = max_steps
        self.done = False
        self._obs_space_updated = False

        self.group_states = []
        self.group_rewards = []
        self.group_actions = []

    def _log(self, message: str, level: str = 'info') -> None:
        if self.verbose >= (1 if level == 'info' else 2):
            getattr(self.logger, level)(message)

    def generate_cats_ids(self, dataset: Optional[pd.DataFrame] = None, 
                         categorical_columns: Optional[List[str]] = None) -> List[Tuple[int, int, np.ndarray]]:
        dataset = dataset or self.tab_dataset
        categorical_columns = categorical_columns or self.categorical_columns
        cat_metadata = []
        cat_maps = {}
        for index, key in enumerate(self.feature_order):
            if key in categorical_columns:
                unique_vals = pd.unique(dataset[key])
                cat_metadata.append((index, len(unique_vals), unique_vals))
                cat_maps[key] = {val: i for i, val in enumerate(unique_vals)}
        self.cat_maps = cat_maps
        return cat_metadata

    def define_action_space(self) -> List[str]:
        action_space = []
        for column in self.feature_order:
            constraint = self.constraints.get(column, None)
            if constraint == "fixed":
                continue
            if column in self.categorical_columns:
                if constraint in [None, "change"]:
                    action_space.append(f'change_{column}')
            else:
                if constraint == "increase":
                    action_space.append(f'increase_{column}')
                elif constraint == "decrease":
                    action_space.append(f'decrease_{column}')
                elif constraint is None:
                    action_space.append(f'increase_{column}')
                    action_space.append(f'decrease_{column}')
        self._log(f"Defined action space with {len(action_space)} actions: {action_space}")
        return action_space

    def apply_action(self, action_idx: int) -> np.ndarray:
        action_name = self._action_idx_to_name(action_idx)
        modified_features = self.modified_features.copy()
        try:
            action_type, feature_name = self._parse_action(action_name)
            feature_index = self.feature_order.index(feature_name)
            if action_type == 'change':
                modified_features = self._apply_categorical_action(feature_index, modified_features)
            elif action_type in ['increase', 'decrease']:
                modified_features = self._apply_continuous_action(feature_index, action_type, modified_features)
        except Exception as e:
            self._log(f"Error applying action {action_name}: {e}", level='error')
        return modified_features

    def _parse_action(self, action_name: str) -> Tuple[str, str]:
        parts = action_name.split('_', 1)
        return parts[0], parts[1]

    def _apply_categorical_action(self, feature_index: int, modified_features: np.ndarray) -> np.ndarray:
        for idx, ncat, cat_values in self.categorical_metadata:
            if idx == feature_index and ncat > 1:
                current_value = modified_features[feature_index]
                available_values = [val for val in cat_values if val != current_value]
                if available_values:
                    modified_features[feature_index] = np.random.choice(available_values)
        return modified_features

    def _apply_continuous_action(self, feature_index: int, action_type: str, 
                               modified_features: np.ndarray) -> np.ndarray:
        current_value = modified_features[feature_index]
        progress = self.steps_taken / self.max_steps
        step_size = np.random.uniform(0.05, 0.25) * (1 - progress) + np.random.uniform(0.05, 0.15) * progress
        
        if action_type == 'increase':
            max_value = self.tab_dataset.iloc[:, feature_index].max()
            calculated_value = current_value * (1 + step_size)
            if self.tab_dataset.iloc[:, feature_index].dtype in [np.int64, np.int32, int]:
                calculated_value = int(calculated_value)
            new_value = min(calculated_value, max_value)
        else:
            min_value = self.tab_dataset.iloc[:, feature_index].min()
            calculated_value = current_value * (1 - step_size)
            if self.tab_dataset.iloc[:, feature_index].dtype in [np.int64, np.int32, int]:
                calculated_value = int(calculated_value)
            new_value = max(calculated_value, min_value)
        modified_features[feature_index] = new_value
        return modified_features

    def _action_idx_to_name(self, action_idx: int) -> str:
        return self.action_names[action_idx]

    def reset(self, instance_idx=None) -> np.ndarray:
        if instance_idx is not None:
            self.current_instance_idx = instance_idx
        else:
            if self.use_random_sampling:
                self.current_instance_idx = np.random.randint(0, len(self.tab_dataset))
            else:
                if self.current_instance_idx is None or self.current_instance_idx < 0 or \
                   self.current_instance_idx >= len(self.tab_dataset):
                    raise ValueError(f"Invalid current_instance_idx: {self.current_instance_idx}")
        
        self.original_features = self.tab_dataset.iloc[self.current_instance_idx].values[:-1]
        self.modified_features = self.original_features.copy()
        self.original_encoded = self.encode_features(self.original_features)
        self.modified_encoded = self.encode_features(self.modified_features)
        self.original_prediction = self.generate_prediction(self.model, self.original_features)

        if not hasattr(self, 'model_output_dim'):
            self.model_output_dim = self._compute_model_output_dim()

        if not self._obs_space_updated:
            full_obs_dim = len(self._get_observation())
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(full_obs_dim,), dtype=np.float32
            )
            self._obs_space_updated = True

        self.steps_taken = 0
        self.done = False
        self.group_states = []
        self.group_rewards = []
        self.group_actions = []
        return self._get_observation()

    def _compute_model_output_dim(self) -> int:
        test_features = self.encode_features(self.original_features)
        with torch.no_grad():
            output = self.model(torch.tensor(test_features, dtype=torch.float32).unsqueeze(0))
        return output.shape[-1]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps_taken += 1
        temperature = self._compute_temperature()
        action_probs = self.get_action_probs(action, temperature)
        action = np.random.choice(self.action_space.n, p=action_probs)
        self.modified_features = self.apply_action(action)
        modified_prediction = self.generate_prediction(self.model, self.modified_features)
        counterfactual_found = (modified_prediction != self.original_prediction)
        self.done = counterfactual_found or (self.steps_taken >= self.max_steps)
        distance = self.calculate_distance(self.original_features, self.modified_features)
        reward = self.calculate_reward(counterfactual_found)
        observation = self._get_observation()

        self.group_states.append(observation)
        self.group_rewards.append(reward)
        self.group_actions.append(action)

        info = {
            'distance': distance,
            'counterfactual_found': counterfactual_found,
            'original_prediction': self.original_prediction,
            'modified_prediction': modified_prediction,
            'steps_taken': self.steps_taken,
            'modified_features': self.modified_features,
            'action_taken': self._action_idx_to_name(action),
            'reward': reward
        }
        return observation, reward, self.done, info

    def _compute_temperature(self) -> float:
        if self.steps_taken < self.RANDOMISATION_THRESHOLD:
            return self.INITIAL_TEMPERATURE
        progress = (self.steps_taken - self.RANDOMISATION_THRESHOLD) / \
                   (self.max_steps - self.RANDOMISATION_THRESHOLD)
        return self.INITIAL_TEMPERATURE + (self.MAX_TEMPERATURE - self.INITIAL_TEMPERATURE) * progress

    def get_action_probs(self, action: int, temperature: float) -> np.ndarray:
        probs = np.ones(self.action_space.n) / self.action_space.n
        probs[action] += 1.0
        probs = np.exp(np.log(probs + 1e-10) / temperature)
        return probs / (np.sum(probs) + 1e-10)

    def _get_observation(self) -> np.ndarray:
        distance = self.calculate_distance(self.original_features, self.modified_features)
        steps_normalized = self.steps_taken / self.max_steps
        self.sample_distance = distance
        target_vector = np.zeros(self.model_output_dim, dtype=np.float32)
        target_vector[self.original_prediction] = 1.0
        observation = np.concatenate([
            self.original_encoded,
            self.modified_encoded,
            [steps_normalized, self.sample_distance],
            target_vector
        ]).astype(np.float32)
        return np.nan_to_num(observation, nan=0.0)

    def calculate_distance(self, original_features: np.ndarray, modified_features: np.ndarray) -> float:
        dist = 0
        for i, (orig, mod) in enumerate(zip(original_features, modified_features)):
            dist += float(orig != mod) if i in self.categorical_indices else abs(orig - mod)
        return dist

    def calculate_reward(self, counterfactual_found: bool) -> float:
        features_encoded = self.encode_features(self.modified_features)
        features_tensor = torch.tensor(features_encoded, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        confidence_reward = 10 - 10 * probs[self.original_prediction]
        counterfactual_bonus = 100.0 if counterfactual_found else 0.0
        return (confidence_reward + counterfactual_bonus)

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        X = np.array(features, dtype=object).reshape(1, -1)
        for i, col in enumerate(self.feature_order):
            if col in self.label_encoders:
                encoder = self.label_encoders[col]
                val = str(X[0, i])
                X[0, i] = encoder.transform([val])[0] if val in encoder.classes_ else 0
        if self.scaler:
            X = self.scaler.transform(X)
        return X.flatten().astype(np.float32)

    def generate_prediction(self, model, features: np.ndarray) -> int:
        try:
            features_encoded = self.encode_features(features)
            features_tensor = torch.tensor(features_encoded, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(features_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            return np.argmax(probs)
        except Exception as e:
            self._log(f"Error in generate_prediction: {e}", level='error')
            return 0

    def get_con_cat_columns(self) -> Tuple[List[str], List[str]]:
        feature_cols = list(self.tab_dataset.columns[:-1])
        continuous_columns = [col for col in feature_cols if self.tab_dataset[col].dtype != 'O']
        categorical_columns = [col for col in feature_cols if self.tab_dataset[col].dtype == 'O']
        return continuous_columns, categorical_columns

    def generate_cats_ids(self, dataset: Optional[pd.DataFrame] = None, cat: Optional[List[str]] = None) -> List[Tuple[int, int, np.ndarray]]:
        dataset = dataset or self.tab_dataset
        cat = cat or self.categorical_columns
        cat_ids = []
        for index, key in enumerate(self.feature_order):
            if key in set(cat):
                cat_ids.append((index, len(pd.unique(dataset[key])), pd.unique(dataset[key])))
        return cat_ids

    def sample_group_trajectories(self, policy_fn, group_size=4, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps
            
        trajectories = []
        initial_state = self.reset()
        initial_idx = self.current_instance_idx
        
        for _ in range(group_size):
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
                try:
                    action, log_prob = policy_fn(obs)
                    action = action.item() if isinstance(action, torch.Tensor) else action
                    log_prob = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
                except Exception:
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
        total_rewards = [traj['total_reward'] for traj in trajectories]
        
        if len(total_rewards) == 0:
            return []
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards) if len(total_rewards) > 1 else 1.0
        
        if std_reward < 1e-8:
            std_reward = 1.0
        
        advantages = [(reward - mean_reward) / std_reward for reward in total_rewards]
        
        return advantages

class GRPOMonitorCallback(BaseCallback):
    def __init__(self, eval_env=None, eval_freq: int = 1000, n_eval_episodes: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.last_eval_time = time.time()
        self.counterfactuals_found = 0
        self.total_episodes = 0
        self.success_rate = 0.0
        self._logger = logging.getLogger(__name__)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            current_time = time.time()
            fps = self.eval_freq / (current_time - self.last_eval_time)
            self.last_eval_time = current_time
            if self.verbose > 1:
                self._logger.debug(f"Steps: {self.n_calls}, FPS: {fps:.2f}")
                self._logger.debug(f"Current success rate: {self.success_rate:.2%}")
        return True