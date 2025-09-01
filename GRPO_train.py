import os
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.distributions import Categorical
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings("ignore")

from GRPO_env import GRPOEnv, GRPOMonitorCallback
from Classifier_model import Classifier, evaluate_model_on_full_dataset, train_model
from closest_samples import get_closest_samples
from KPIs import proximity_KPI, sparsity_KPI

logging.basicConfig(level=logging.INFO)

class Config:
    """Configuration constants for GRPO training and counterfactual generation."""
    DATASET_NAME: str = 'bank'
    DATASET_PATH: str = os.path.join('data', f'{DATASET_NAME}.csv')
    TOTAL_ITERATIONS: int = 10
    STEPS_PER_ITERATION: int = 10
    MU_ITERATIONS: int = 4
    GROUP_SIZE: int = 4
    LEARNING_RATE: float = 3e-4
    CLIP_RANGE: float = 0.2
    KL_COEF: float = 0.01
    MAX_GRAD_NORM: float = 0.5
    CONSTRAINTS: Dict[str, str] = {}
    SAVE_DIR: str = 'grpo_models'
    DATA_DIR: str = 'data'
    LOGS_DIR: str = 'grpo_logs'
    MAX_STEPS_PER_SAMPLE: int = 250
    MAX_TRIES: int = 50
    INDICES_TO_USE: Optional[List[int]] = list(range(25))
    TRAINING_MODE: str = 'new'  # 'new' or 'continue'

class GRPOPolicy(nn.Module):
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
        logits = self.network(obs)
        return logits
    
    def get_action_and_log_prob(self, obs, deterministic=False):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def get_log_prob(self, obs, action):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)

class GRPOTrainer:
    def __init__(self, env, policy, learning_rate=Config.LEARNING_RATE, clip_range=Config.CLIP_RANGE, 
                 kl_coef=Config.KL_COEF, group_size=Config.GROUP_SIZE, max_grad_norm=Config.MAX_GRAD_NORM, 
                 mu_iterations=Config.MU_ITERATIONS):
        self.env = env
        self.policy = policy
        self.group_size = group_size
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.mu_iterations = mu_iterations
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        self.reference_policy = self._create_policy_copy()
        self.reference_policy.eval()
        
        self.old_policy = None
        
        self.training_stats = {
            'policy_loss': [],
            'kl_divergence': [],
            'success_rate': [],
            'mean_reward': [],
            'mean_advantage': []
        }
    
    def _create_policy_copy(self):
        policy_copy = GRPOPolicy(
            obs_dim=self.policy.network[0].in_features,
            action_dim=self.policy.network[-1].out_features,
            hidden_dim=256
        )
        policy_copy.load_state_dict(self.policy.state_dict())
        return policy_copy
    
    def train_step(self, num_episodes=10):
        self.old_policy = self._create_policy_copy()
        self.old_policy.eval()
        
        all_trajectories = []
        episode_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            trajectories = self.env.sample_group_trajectories(
                policy_fn=lambda obs: self.old_policy.get_action_and_log_prob(
                    torch.FloatTensor(obs).unsqueeze(0), deterministic=False
                ),
                group_size=self.group_size
            )
            
            all_trajectories.extend(trajectories)
            
            for traj in trajectories:
                episode_rewards.append(traj['total_reward'])
                if traj['counterfactual_found']:
                    success_count += 1
        
        advantages = self.env.compute_group_advantages(all_trajectories)
        
        total_policy_loss = 0
        total_kl_div = 0
        
        for _ in range(self.mu_iterations):
            policy_loss, kl_div = self._compute_grpo_loss(all_trajectories, advantages)
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_kl_div += kl_div
        
        avg_policy_loss = total_policy_loss / self.mu_iterations
        avg_kl_div = total_kl_div / self.mu_iterations
        
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
    
    def _compute_grpo_loss(self, trajectories, advantages):
        total_loss = 0
        total_kl = 0
        num_trajectories = 0
        
        for traj_idx, (trajectory, advantage) in enumerate(zip(trajectories, advantages)):
            if len(trajectory['states']) == 0:
                continue
                
            states = torch.FloatTensor(np.array(trajectory['states']))
            actions = torch.LongTensor(trajectory['actions'])
            old_log_probs = torch.FloatTensor(trajectory['log_probs'])
            
            current_log_probs = self.policy.get_log_prob(states, actions)
            
            with torch.no_grad():
                ref_log_probs = self.reference_policy.get_log_prob(states, actions)
            
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            advantage_tensor = torch.tensor(advantage, dtype=torch.float32)
            surrogate1 = ratio * advantage_tensor
            surrogate2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage_tensor
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            kl_div = (current_log_probs - ref_log_probs).mean()
            
            total_loss += policy_loss + self.kl_coef * kl_div
            total_kl += kl_div.item()
            num_trajectories += 1
        
        if num_trajectories == 0:
            return torch.tensor(0.0, requires_grad=True), 0.0
            
        return total_loss / num_trajectories, total_kl / num_trajectories
    
    def update_reference_policy(self):
        self.reference_policy.load_state_dict(self.policy.state_dict())

def setup_directories(directories: List[str]) -> None:
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_classifier_model(dataset_path: str, model_path: Optional[str], num_features: int, 
                         num_classes: int, verbose: int = 1) -> Classifier:
    logger = logging.getLogger(__name__)
    if model_path is None:
        model_path = os.path.join("classification_models", 
                                 f"{os.path.splitext(os.path.basename(dataset_path))[0]}_model.pt")
    
    if not os.path.exists(model_path):
        logger.info(f"Classifier model not found at {model_path}. Training a new one.")
        train_model(dataset_path, model_path)
    
    logger.info(f"Loading classifier model from {model_path}")
    classifier = Classifier(
        in_feats=num_features,
        out=num_classes,
        h_size=128,
        n_layers=4,
        dropout=0.3,
        lr=0.001,
        weight_decay=1e-4
    )
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    accuracy = evaluate_model_on_full_dataset(classifier, dataset_path)
    logger.info(f"Classifier accuracy on full dataset: {accuracy:.2f}")
    return classifier

def load_encoders_and_scaler(model_path: str) -> Tuple[Optional[Dict], Optional[object]]:
    logger = logging.getLogger(__name__)
    encoders_path = model_path.replace('.pt', '_encoders.pkl')
    scaler_path = model_path.replace('.pt', '_scaler.pkl')
    label_encoders, scaler = None, None
    
    try:
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
            logger.info(f"Label encoders loaded from {encoders_path}")
    except Exception as e:
        logger.warning(f"Failed to load label encoders: {e}")
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        logger.warning(f"Failed to load scaler: {e}")
    
    return label_encoders, scaler

def train_grpo_for_counterfactuals(dataset_path: str, model_path: Optional[str] = None, 
                                  save_dir: str = Config.SAVE_DIR,
                                  total_iterations: int = Config.TOTAL_ITERATIONS, 
                                  steps_per_iteration: int = Config.STEPS_PER_ITERATION,
                                  mu_iterations: int = Config.MU_ITERATIONS,
                                  constraints: Optional[Dict[str, str]] = None,
                                  verbose: int = 1,
                                  continue_training: bool = False
                                  ) -> Tuple[GRPOTrainer, GRPOEnv]:
    logger = logging.getLogger(__name__)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    logger.info(f"Starting counterfactual generation with GRPO for dataset: {dataset_name}")
    
    setup_directories([Config.DATA_DIR, Config.LOGS_DIR, Config.SAVE_DIR])
    
    dataset_utils = DatasetUtils(dataset_path, verbose)
    classifier = load_classifier_model(dataset_path, model_path, dataset_utils.num_features, 
                                      dataset_utils.num_classes, verbose)
    label_encoders, scaler = load_encoders_and_scaler(model_path or 
                                                     f"classification_models/{dataset_name}_model.pt")
    
    max_steps = Config.MAX_STEPS_PER_SAMPLE
    
    env = GRPOEnv(dataset_path=dataset_path, model=classifier, label_encoders=label_encoders, 
                 scaler=scaler, constraints=constraints, use_random_sampling=True, max_steps=max_steps, verbose=verbose)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = GRPOPolicy(obs_dim, action_dim)
    
    policy_model_path = os.path.join(save_dir, f"grpo_policy_{dataset_name}_final.pt")
    
    model_exists = os.path.exists(policy_model_path)
    if model_exists and continue_training:
        logger.info(f"Loading existing GRPO policy from {policy_model_path}")
        try:
            policy.load_state_dict(torch.load(policy_model_path))
            logger.info("Existing GRPO policy loaded successfully. Continuing training...")
        except Exception as e:
            logger.warning(f"Error loading existing model: {e}. Creating a new GRPO policy...")
            model_exists = False
    
    trainer = GRPOTrainer(
        env=env,
        policy=policy,
        mu_iterations=mu_iterations
    )
    
    logger.info(f"Starting GRPO training...")
    
    for iteration in range(total_iterations):
        trainer.update_reference_policy()
        
        logger.info(f"\n=== Iteration {iteration+1}/{total_iterations} ===")
        
        for step in range(steps_per_iteration):
            step_stats = trainer.train_step(num_episodes=5)
            
            if step % 1 == 0:
                logger.info(f"  Step {step+1}/{steps_per_iteration}: "
                      f"Loss={step_stats['policy_loss']:.4f}, "
                      f"Success={step_stats['success_rate']:.2%}, "
                      f"μ_iters={step_stats['grpo_iterations']}")
        
        recent_stats = trainer.training_stats
        if recent_stats['policy_loss']:
            avg_loss = np.mean(recent_stats['policy_loss'][-steps_per_iteration:])
            avg_success = np.mean(recent_stats['success_rate'][-steps_per_iteration:])
            avg_reward = np.mean(recent_stats['mean_reward'][-steps_per_iteration:])
            
            logger.info(f"Iteration {iteration+1} Summary:")
            logger.info(f"  Average Loss: {avg_loss:.4f}")
            logger.info(f"  Average Success Rate: {avg_success:.2%}")
            logger.info(f"  Average Reward: {avg_reward:.2f}")
        
        if (iteration + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"grpo_policy_iter_{iteration+1}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    final_path = os.path.join(save_dir, f"grpo_policy_final.pt")
    torch.save(policy.state_dict(), final_path)
    logger.info(f"Final model saved: {final_path}")
    
    return trainer, env

def generate_counterfactuals(trainer: GRPOTrainer, env: GRPOEnv, dataset_path: str, 
                            save_path: Optional[str] = None, specific_indices: Optional[List[int]] = None,
                            max_steps_per_sample: int = Config.MAX_STEPS_PER_SAMPLE, 
                            verbose: int = 1) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    logger = logging.getLogger(__name__)
    logger.info("Generating counterfactuals using GRPO...")
    
    dataset_utils = DatasetUtils(dataset_path, verbose)
    env.use_random_sampling = False
    
    indices_to_use = specific_indices or list(range(len(dataset_utils.dataset)))
    logger.info(f"Processing {len(indices_to_use)} samples")
    
    counterfactuals, original_samples, counterfactual_samples = [], [], []
    success_count, total_steps = 0, 0
    start_time = time.time()
    result_saver = ResultSaver(verbose)
    
    trainer.policy.eval()
    
    for i, idx in tqdm(enumerate(indices_to_use), total=len(indices_to_use), desc="Generating counterfactuals"):
        env.current_instance_idx = idx
        success, best_info, steps_taken = generate_single_counterfactual(
            env, trainer.policy, max_steps_per_sample, Config.MAX_TRIES, verbose)
        
        original_features = env.original_features
        original_prediction = env.original_prediction
        counterfactuals.append({
            'sample_id': i,
            'data_index': idx,
            'original_features': original_features,
            'counterfactual_features': best_info['modified_features'] if best_info else None,
            'original_prediction': original_prediction,
            'counterfactual_prediction': best_info['modified_prediction'] if best_info else original_prediction,
            'distance': best_info['distance'] if best_info else float('inf'),
            'steps': steps_taken,
            'tries': Config.MAX_TRIES,
            'success': success
        })
        original_samples.append(original_features)
        counterfactual_samples.append(best_info['modified_features'] if best_info else original_features)
        success_count += success
        total_steps += steps_taken
    
    original_df = pd.DataFrame(original_samples, columns=dataset_utils.feature_columns)
    counterfactual_df = pd.DataFrame(counterfactual_samples, columns=dataset_utils.feature_columns)
    
    if save_path and counterfactuals:
        result_saver.save_counterfactuals_to_csv(counterfactuals, original_df, counterfactual_df, save_path)
    
    env.use_random_sampling = True
    return counterfactuals, original_df, counterfactual_df

def generate_single_counterfactual(env: GRPOEnv, policy: GRPOPolicy, 
                                  max_steps: int, max_tries: int, 
                                  verbose: int) -> Tuple[bool, Optional[Dict], int]:
    logger = logging.getLogger(__name__)
    best_info = None
    best_distance = float('inf')
    success = False
    total_steps = 0
    
    for tries in range(max_tries):
        obs = env.reset()
        done, steps = False, 0
        while not done and steps < max_steps:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _ = policy.get_action_and_log_prob(obs_tensor, deterministic=True)
                action = action.item()
            obs, reward, done, info = env.step(action)
            steps += 1
            total_steps += 1
            if info['counterfactual_found']:
                current_distance = info['distance']
                if current_distance < best_distance:
                    best_distance = current_distance
                    best_info = info
                    success = True
    
    return success, best_info, total_steps

def calculate_distance(original_features: np.ndarray, modified_features: np.ndarray) -> float:
    categorical_indices = [i for i, feature in enumerate(original_features) 
                          if isinstance(feature, (str, bool))]
    dist = 0
    for i, (o, m) in enumerate(zip(original_features, modified_features)):
        dist += float(o != m) if i in categorical_indices else abs(o - m)
    return dist

def get_metrics(original_df: pd.DataFrame, counterfactual_df: pd.DataFrame, counterfactuals: List[Dict],
                constraints: Dict[str, str], feature_columns: List[str], original_data: pd.DataFrame,
                verbose: int = 1) -> Tuple[float, float, float, float, float]:
    logger = logging.getLogger(__name__)
    logger.debug(f"Calculating KPIs for {len(counterfactuals)} counterfactuals...")
    
    if not counterfactuals:
        logger.warning("No counterfactuals generated. Returning default KPIs.")
        return 0, 0, 0, 0, 0
    
    success_count = sum(cf['success'] for cf in counterfactuals)
    num_samples = len(counterfactuals)
    coverage = success_count / num_samples if num_samples > 0 else 0
    logger.debug(f"Coverage: {coverage:.2%} ({success_count}/{num_samples})")
    
    distance = np.mean([cf['distance'] for cf in counterfactuals]) if counterfactuals else 0
    logger.debug(f"Mean distance of counterfactuals: {distance:.2f}")
    
    implausibility_scores = []
    for cf in counterfactuals:
        if cf['counterfactual_features'] is not None:
            cfe = np.concatenate((cf['counterfactual_features'], [cf['counterfactual_prediction']]))
            closest_sample = get_closest_samples(cfe, original_data, X=5, require_different_outcome=False).iloc[0]
            implausibility_scores.append(calculate_distance(cfe[:-1], closest_sample[:-1]))
    implausibility = np.mean(implausibility_scores) if implausibility_scores else 0
    logger.debug(f"Mean implausibility of counterfactuals: {implausibility:.2f}")
    
    sparsity = 0
    if len(original_df) > 0 and len(counterfactual_df) > 0:
        assert len(original_df) == len(counterfactual_df), \
            "Original and counterfactual DataFrames must have the same length"
        assert original_df.columns.tolist() == counterfactual_df.columns.tolist(), \
            "Original and counterfactual DataFrames must have the same columns"
        
        sparsities = []
        for i in range(len(original_df)):
            if counterfactuals[i]['counterfactual_features'] is not None:
                changes = sum(1 for col in original_df.columns 
                             if original_df[col].iloc[i] != counterfactual_df[col].iloc[i])
                sparsities.append(changes)
        sparsity = round(sum(sparsities) / len(sparsities), 4) if sparsities else 0
    logger.debug(f"Sparsity KPI: {sparsity}")
    
    actionability_scores = [
        calculate_actionability(cf['original_features'], cf['counterfactual_features'], 
                              constraints, feature_columns)
        for cf in counterfactuals if cf['counterfactual_features'] is not None
    ]
    actionability = np.mean(actionability_scores) if actionability_scores else 0
    logger.debug(f"Mean actionability of counterfactuals: {actionability:.2f}")
    
    return coverage, distance, implausibility, sparsity, actionability

def calculate_actionability(original_features: np.ndarray, counterfactual_features: np.ndarray, 
                           constraints: Dict[str, str], feature_columns: List[str]) -> int:
    if not constraints:
        return 1
    for feature, constraint in constraints.items():
        if constraint == "fixed":
            idx = feature_columns.index(feature)
            if original_features[idx] != counterfactual_features[idx]:
                return 0
    return 1

class DatasetUtils:
    def __init__(self, dataset_path: str, verbose: int = 1):
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.dataset = self._load_dataset(dataset_path)
        self.num_features = len(self.dataset.columns) - 1
        self.num_classes = len(self.dataset.iloc[:, -1].unique())
        self.feature_columns = self.dataset.columns[:-1].tolist()

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        dataset = pd.read_csv(dataset_path)
        self._log(f"Loaded dataset from {dataset_path} with {len(dataset)} samples")
        return dataset

    def _log(self, message: str, level: str = 'info') -> None:
        if self.verbose >= (1 if level == 'info' else 2):
            getattr(self.logger, level)(message)

class ResultSaver:
    def __init__(self, verbose: int = 1):
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

    def save_counterfactuals_to_csv(self, counterfactuals: List[Dict], original_df: pd.DataFrame, 
                                   counterfactual_df: pd.DataFrame, save_path: str) -> None:
        counterfactual_rows = []
        original_rows = []
        
        feature_columns = original_df.columns.tolist()
        
        for i, cf in enumerate(counterfactuals):
            sample_id = cf['sample_id']
            original_row = {col: original_df.iloc[i][col] for col in feature_columns}
            original_row['sample_id'] = sample_id
            original_rows.append(original_row)
            
            if cf['counterfactual_features'] is not None and cf['success']:
                row = {col: counterfactual_df.iloc[i][col] for col in feature_columns}
                row['sample_id'] = sample_id
                row['counterfactual_found'] = 1
            else:
                row = {col: original_df.iloc[i][col] for col in feature_columns}
                row['sample_id'] = sample_id
                row['counterfactual_found'] = 0
            counterfactual_rows.append(row)
        
        counterfactual_df_out = pd.DataFrame(counterfactual_rows)
        original_df_out = pd.DataFrame(original_rows)
        
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        counterfactual_df_out.to_csv(save_path, index=False)
        original_filename = f"{os.path.splitext(save_path)[0]}_original.csv"
        original_df_out.to_csv(original_filename, index=False)
        
        self._log(f"Counterfactuals saved to {save_path}")
        self._log(f"Original samples saved to {original_filename}")

    def _log(self, message: str, level: str = 'info') -> None:
        if self.verbose >= (1 if level == 'info' else 2):
            getattr(self.logger, level)(message)

def main():
    logger = logging.getLogger(__name__)
    setup_directories([Config.DATA_DIR, Config.LOGS_DIR, Config.SAVE_DIR])
    
    dataset_path = Config.DATASET_PATH
    constraints = Config.CONSTRAINTS
    indices_to_use = Config.INDICES_TO_USE
    
    logger.info("Processing GRPO model...")
    trainer, env = train_grpo_for_counterfactuals(
        dataset_path=dataset_path,
        save_dir=Config.SAVE_DIR,
        total_iterations=Config.TOTAL_ITERATIONS,
        steps_per_iteration=Config.STEPS_PER_ITERATION,
        mu_iterations=Config.MU_ITERATIONS,
        constraints=constraints,
        verbose=1
    )
    
    if trainer is not None:
        logger.info("Generating counterfactuals using GRPO...")
        generate_counterfactuals(
            trainer=trainer,
            env=env,
            dataset_path=dataset_path,
            save_path=os.path.join(Config.DATA_DIR, 
                                  f"generated_counterfactuals_grpo_{os.path.splitext(os.path.basename(dataset_path))[0]}.csv"),
            max_steps_per_sample=Config.MAX_STEPS_PER_SAMPLE,
            specific_indices=indices_to_use,
            verbose=1
        )

if __name__ == "__main__":
    main()