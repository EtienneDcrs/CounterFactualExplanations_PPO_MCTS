import os
import pickle
import time
import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
from closest_samples import get_closest_samples
from PPO_env import PPOEnv, PPOMonitorCallback
from Classifier_model import Classifier, evaluate_model_on_full_dataset, train_model
from PPO_MCTS import PPOMCTS
# Import GRPO components
from GRPO_env import GRPOEnv
from GRPO_train import GRPOTrainer, GRPOPolicy
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)

class Config:
    """Configuration constants for PPO training and counterfactual generation."""
    DATASET_NAME: str = 'adult'
    DATASET_PATH: str = os.path.join('data', f'{DATASET_NAME}.csv')
    TOTAL_TIMESTEPS: int = 75000
    # Types of constraints : increase, decrease (only for numerical features), fixed (any feature)
    CONSTRAINTS: Dict[str, str] =  {}
    SAVE_DIR: str = 'ppo_models'
    DATA_DIR: str = 'data'
    LOGS_DIR: str = 'ppo_logs'
    CHECKPOINT_PREFIX: str = 'ppo'
    LEARNING_RATE_NEW: float = 1e-4
    LEARNING_RATE_CONTINUED: float = 1e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 128
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    ENT_COEF: float = 0.05
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    EVAL_FREQ: int = 5000
    N_EVAL_EPISODES: int = 10
    CHECKPOINT_SAVE_FREQ: int = 50000
    MAX_STEPS_PER_SAMPLE: int = 150
    MAX_TRIES: int = 50
    MCTS_SIMULATIONS: int = 10
    # Use a random selection of 100 indices from the dataset for evaluation
    INDICES_TO_USE: Optional[List[int]] = list(range(100)) #np.random.choice(200, size=100, replace=False).tolist()
    # Training mode options: 'new', 'load', or 'continue'
    TRAINING_MODE: str = 'new'
    GRPO_TOTAL_ITERATIONS: int = 10
    GRPO_STEPS_PER_ITERATION: int = 10
    GRPO_MU_ITERATIONS: int = 4
    GRPO_GROUP_SIZE: int = 4
    GRPO_KL_COEF: float = 0.01

class DatasetUtils:
    """Utility class for dataset handling."""
    def __init__(self, dataset_path: str, verbose: int = 1):
        """
        Initialize dataset utilities.

        Args:
            dataset_path: Path to the dataset CSV file.
            verbose: Verbosity level for logging (0: none, 1: info, 2: debug).
        """
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.dataset = self._load_dataset(dataset_path)
        self.num_features = len(self.dataset.columns) - 1
        self.num_classes = len(self.dataset.iloc[:, -1].unique())
        self.feature_columns = self.dataset.columns[:-1].tolist()

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load and validate dataset."""
        dataset = pd.read_csv(dataset_path)
        self._log(f"Loaded dataset from {dataset_path} with {len(dataset)} samples")
        return dataset

    def _log(self, message: str, level: str = 'info') -> None:
        """Log messages based on verbosity level."""
        if self.verbose >= (1 if level == 'info' else 2):
            getattr(self.logger, level)(message)

class ResultSaver:
    """Utility class for saving counterfactual results."""
    def __init__(self, verbose: int = 1):
        """
        Initialize result saver.

        Args:
            verbose: Verbosity level for logging (0: none, 1: info, 2: debug).
        """
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

    def save_counterfactuals(self, counterfactuals: List[Dict], original_data: pd.DataFrame, 
                            save_path: str, multiple: bool = False) -> None:
        """
        Save counterfactuals to CSV.

        Args:
            counterfactuals: List of counterfactual dictionaries.
            original_data: Original dataset for column names.
            save_path: Path to save the CSV file.
            multiple: Whether to include counterfactual_id for multiple counterfactuals per sample.
        """
        counterfactual_data = []
        for cf in counterfactuals:
            row = {
                'sample_id': cf['sample_id'],
                'data_index': cf['data_index'],
                'original_prediction': cf['original_prediction'],
                'counterfactual_prediction': cf['counterfactual_prediction'],
                'distance': cf['distance'],
                'steps': cf['steps']
            }
            if multiple:
                row['counterfactual_id'] = cf['counterfactual_id']
            for i, col in enumerate(original_data.columns[:-1]):
                row[f'original_{col}'] = cf['original_features'][i]
                row[f'counterfactual_{col}'] = cf['counterfactual_features'][i]
            counterfactual_data.append(row)
        
        df = pd.DataFrame(counterfactual_data)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        df.to_csv(save_path, index=False)
        self._log(f"Saved counterfactuals to {save_path}")

    def save_counterfactuals_to_csv(self, counterfactuals: List[Dict], original_df: pd.DataFrame, 
                                   counterfactual_df: pd.DataFrame, save_path: str) -> None:
        """
        Save counterfactuals and original samples to CSV files in a format compatible with utils.get_metrics.
        
        Args:
            counterfactuals: List of counterfactual dictionaries containing sample_id, counterfactual_features, etc.
            original_df: DataFrame of original samples.
            counterfactual_df: DataFrame of counterfactual samples.
            save_path: Path to save the counterfactual CSV file (original CSV will append '_original').
        """
        counterfactual_rows = []
        original_rows = []
        
        feature_columns = original_df.columns.tolist()
        
        for i, cf in enumerate(counterfactuals):
            sample_id = cf['sample_id']
            # Save original sample
            original_row = {col: original_df.iloc[i][col] for col in feature_columns}
            original_row['sample_id'] = sample_id
            original_rows.append(original_row)
            
            # Save counterfactual or original features if no counterfactual
            if cf['counterfactual_features'] is not None and cf['success']:
                row = {col: counterfactual_df.iloc[i][col] for col in feature_columns}
                row['sample_id'] = sample_id
                row['counterfactual_found'] = 1
            else:
                row = {col: original_df.iloc[i][col] for col in feature_columns}
                row['sample_id'] = sample_id
                row['counterfactual_found'] = 0
            counterfactual_rows.append(row)
        
        # Create DataFrames
        counterfactual_df_out = pd.DataFrame(counterfactual_rows)
        original_df_out = pd.DataFrame(original_rows)
        
        # Save to CSVs
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        counterfactual_df_out.to_csv(save_path, index=False)
        original_filename = f"{os.path.splitext(save_path)[0]}_original.csv"
        original_df_out.to_csv(original_filename, index=False)
        
        self._log(f"Counterfactuals saved to {save_path}")
        self._log(f"Original samples saved to {original_filename}")

    def _log(self, message: str, level: str = 'info') -> None:
        """Log messages based on verbosity level."""
        if self.verbose >= (1 if level == 'info' else 2):
            getattr(self.logger, level)(message)

def setup_directories(directories: List[str]) -> None:
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_classifier_model(dataset_path: str, model_path: Optional[str], num_features: int, 
                         num_classes: int, verbose: int = 1) -> Classifier:
    """
    Load or train a classifier model.

    Args:
        dataset_path: Path to the dataset CSV file.
        model_path: Path to the pre-trained classifier model.
        num_features: Number of input features.
        num_classes: Number of output classes.
        verbose: Verbosity level for logging.

    Returns:
        Loaded or trained classifier model.
    """
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
    """
    Load label encoders and scaler.

    Args:
        model_path: Path to the classifier model.

    Returns:
        Tuple of (label_encoders, scaler).
    """
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

def initialize_ppo_model(ppo_model_path: str, env: PPOEnv, logs_dir: str, 
                         mode: str, dataset_name: str, verbose: int) -> PPO:
    """
    Initialize or load a PPO model based on the specified mode.

    Args:
        ppo_model_path: Path to the PPO model.
        env: Training environment.
        logs_dir: Directory for tensorboard logs.
        mode: Training mode ('new', 'load', or 'continue').
        dataset_name: Name of the dataset to ensure correct model loading.
        verbose: Verbosity level for logging.

    Returns:
        Initialized or loaded PPO model.

    Raises:
        ValueError: If mode is invalid or model not found for 'load' mode.
    """
    logger = logging.getLogger(__name__)
    valid_modes = ['new', 'load', 'continue']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
    
    model_exists = os.path.exists(ppo_model_path)
    
    if mode == 'new' or not model_exists:
        logger.info("Creating a new PPO agent...")
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=Config.LEARNING_RATE_NEW,
            n_steps=Config.N_STEPS,
            batch_size=Config.BATCH_SIZE,
            n_epochs=Config.N_EPOCHS,
            gamma=Config.GAMMA,
            gae_lambda=Config.GAE_LAMBDA,
            clip_range=Config.CLIP_RANGE,
            clip_range_vf=None,
            ent_coef=Config.ENT_COEF,
            vf_coef=Config.VF_COEF,
            max_grad_norm=Config.MAX_GRAD_NORM,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=logs_dir,
            verbose=verbose
        )
    
    if mode == 'load':
        if not model_exists:
            raise ValueError(f"No PPO model found at {ppo_model_path} for dataset {dataset_name} for loading")
        logger.info(f"Loading existing PPO model from {ppo_model_path} without further training")
        model = PPO.load(ppo_model_path, env=env, tensorboard_log=logs_dir)
        logger.info("PPO model loaded successfully")
        return model
    
    # mode == 'continue'
    logger.info(f"Loading existing PPO model from {ppo_model_path} for continued training")
    try:
        model = PPO.load(
            ppo_model_path,
            env=env,
            tensorboard_log=logs_dir,
            custom_objects={"learning_rate": Config.LEARNING_RATE_CONTINUED}
        )
        model.learning_rate = Config.LEARNING_RATE_CONTINUED
        logger.info(f"Existing PPO model loaded successfully. Learning rate set to {model.learning_rate}")
        return model
    except Exception as e:
        logger.warning(f"Error loading existing model: {e}. Creating a new PPO model...")
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=Config.LEARNING_RATE_NEW,
            n_steps=Config.N_STEPS,
            batch_size=Config.BATCH_SIZE,
            n_epochs=Config.N_EPOCHS,
            gamma=Config.GAMMA,
            gae_lambda=Config.GAE_LAMBDA,
            clip_range=Config.CLIP_RANGE,
            clip_range_vf=None,
            ent_coef=Config.ENT_COEF,
            vf_coef=Config.VF_COEF,
            max_grad_norm=Config.MAX_GRAD_NORM,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=logs_dir,
            verbose=verbose
        )

def train_ppo_for_counterfactuals(dataset_path: str, model_path: Optional[str] = None, 
                                  logs_dir: str = Config.LOGS_DIR, save_dir: str = Config.SAVE_DIR,
                                  total_timesteps: int = Config.TOTAL_TIMESTEPS, 
                                  mode: str = Config.TRAINING_MODE, 
                                  constraints: Optional[Dict[str, str]] = None,
                                  verbose: int = 1,
                                  output_ppo_model_path: Optional[str] = None
                                  ) -> Tuple[PPO, PPOEnv]:
    """
    Train or load a PPO agent for counterfactual generation.

    Args:
        dataset_path: Path to the dataset CSV file.
        model_path: Path to the pre-trained classifier model.
        logs_dir: Directory for storing tensorboard logs.
        save_dir: Directory for saving PPO model checkpoints.
        total_timesteps: Total number of training timesteps.
        mode: Training mode ('new', 'load', or 'continue').
        constraints: Dictionary of feature constraints (e.g., {"age": "increase"}).
        verbose: Verbosity level for logging.
        output_ppo_model_path: Optional path where the PPO model will be saved.

    Returns:
        Tuple of (PPO model, training environment).

    Raises:
        ValueError: If mode is invalid or model not found for 'load' mode.
    """
    logger = logging.getLogger(__name__)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    logger.info(f"Starting counterfactual generation with PPO for dataset: {dataset_name} (mode: {mode})")
    
    setup_directories([logs_dir, save_dir, Config.DATA_DIR])
    
    dataset_utils = DatasetUtils(dataset_path, verbose)
    classifier = load_classifier_model(dataset_path, model_path, dataset_utils.num_features, 
                                      dataset_utils.num_classes, verbose)
    label_encoders, scaler = load_encoders_and_scaler(model_path or 
                                                     f"classification_models/{dataset_name}_model.pt")
    
    max_steps = Config.MAX_STEPS_PER_SAMPLE
    
    env = PPOEnv(dataset_path=dataset_path, model=classifier, label_encoders=label_encoders, 
                 scaler=scaler, constraints=constraints, use_random_sampling=True, max_steps=max_steps, verbose=verbose)
    eval_env = PPOEnv(dataset_path=dataset_path, model=classifier, label_encoders=label_encoders, 
                      scaler=scaler, constraints=constraints, use_random_sampling=True, max_steps=max_steps,verbose=verbose)
    
    # Use output_ppo_model_path if provided, else default
    ppo_model_path = output_ppo_model_path or os.path.join(save_dir, f"{Config.CHECKPOINT_PREFIX}_{dataset_name}_final.zip")
    model = initialize_ppo_model(ppo_model_path, env, logs_dir, mode, dataset_name, verbose)
    
    if mode != 'load':
        checkpoint_callback = CheckpointCallback(
            save_freq=Config.CHECKPOINT_SAVE_FREQ,
            save_path=save_dir,
            name_prefix=f"{Config.CHECKPOINT_PREFIX}_{dataset_name}_checkpoint"
        )
        monitor_callback = PPOMonitorCallback(
            eval_env=eval_env,
            eval_freq=Config.EVAL_FREQ,
            n_eval_episodes=Config.N_EVAL_EPISODES,
            verbose=verbose
        )
        
        logger.info(f"{'Continuing' if mode == 'continue' and os.path.exists(ppo_model_path) else 'Starting new'} "
                    f"training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, monitor_callback],
            reset_num_timesteps=(mode != 'continue' or not os.path.exists(ppo_model_path))
        )
        
        model.save(ppo_model_path)
        logger.info(f"Final model saved to {ppo_model_path}")
    
    logger.info("Evaluating the policy...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model, env

def generate_counterfactuals(ppo_model: PPO, env: PPOEnv, dataset_path: str, 
                            save_path: Optional[str] = None, specific_indices: Optional[List[int]] = None,
                            max_steps_per_sample: int = Config.MAX_STEPS_PER_SAMPLE, 
                            batch_size: Optional[int] = None, use_mcts: bool = False,
                            mcts_simulations: int = Config.MCTS_SIMULATIONS, 
                            verbose: int = 1) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """
    Generate counterfactuals using a trained PPO model.

    Args:
        ppo_model: Trained PPO model.
        env: Environment for counterfactual generation.
        dataset_path: Path to the dataset CSV file.
        save_path: Path to save the generated counterfactuals.
        specific_indices: List of specific data point indices to process.
        max_steps_per_sample: Maximum steps per counterfactual.
        batch_size: Number of samples to process before saving intermediate results.
        use_mcts: Whether to use MCTS for action selection.
        mcts_simulations: Number of MCTS simulations per step.
        verbose: Verbosity level for logging.

    Returns:
        Tuple of (counterfactuals, original_df, counterfactual_df).
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating counterfactuals...")
    feature_dim = env.observation_space.shape[0]
    print(f"Feature dimension: {feature_dim}")
    
    dataset_utils = DatasetUtils(dataset_path, verbose)
    env.use_random_sampling = False
    mcts = PPOMCTS(env=env, ppo_model=ppo_model, num_simulations=mcts_simulations) if use_mcts else None
    if use_mcts:
        logger.info(f"Initialized MCTS with {mcts_simulations} simulations per step")
    
    indices_to_use = specific_indices or list(range(len(dataset_utils.dataset)))
    logger.info(f"Processing {len(indices_to_use)} samples")
    
    counterfactuals, original_samples, counterfactual_samples = [], [], []
    success_count, total_steps = 0, 0
    start_time = time.time()
    result_saver = ResultSaver(verbose)
    
    for i, idx in tqdm(enumerate(indices_to_use), total=len(indices_to_use), desc="Generating counterfactuals"):
        env.current_instance_idx = idx
        success, best_info, steps_taken = generate_single_counterfactual(
            env, ppo_model, mcts, max_steps_per_sample, Config.MAX_TRIES, use_mcts, verbose)
        
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
        
        if batch_size and (i + 1) % batch_size == 0:
            intermediate_save_path = f"{os.path.splitext(save_path)[0]}_batch_{(i+1)//batch_size}.csv"
            result_saver.save_counterfactuals(counterfactuals, dataset_utils.dataset, intermediate_save_path)
            logger.info(f"Saved intermediate results to {intermediate_save_path}")
    
    original_df = pd.DataFrame(original_samples, columns=dataset_utils.feature_columns)
    counterfactual_df = pd.DataFrame(counterfactual_samples, columns=dataset_utils.feature_columns)
    
    if save_path and counterfactuals:
        result_saver.save_counterfactuals_to_csv(counterfactuals, original_df, counterfactual_df, save_path)
    
    env.use_random_sampling = True
    return counterfactuals, original_df, counterfactual_df

def generate_single_counterfactual(env: PPOEnv, ppo_model: PPO, mcts: Optional[PPOMCTS], 
                                  max_steps: int, max_tries: int, use_mcts: bool, 
                                  verbose: int) -> Tuple[bool, Optional[Dict], int]:
    """
    Generate a single counterfactual for a sample.

    Args:
        env: Environment for counterfactual generation.
        ppo_model: Trained PPO model.
        mcts: MCTS instance for action selection.
        max_steps: Maximum steps per try.
        max_tries: Maximum number of tries.
        use_mcts: Whether to use MCTS.
        verbose: Verbosity level for logging.

    Returns:
        Tuple of (success, best_info, total_steps).
    """
    logger = logging.getLogger(__name__)
    best_info = None
    best_distance = float('inf')
    success = False
    total_steps = 0
    
    for tries in range(max_tries):
        if success:# and tries > 20:
            #logger.info(f"Found a successful counterfactual after {tries} tries, stopping further attempts.")
            break
        obs = env.reset()
        done, steps = False, 0
        while not done and steps < max_steps:
            action = mcts.run_mcts(root_state=obs, temperature=0.5) if use_mcts else \
                    ppo_model.predict(obs, deterministic=True)[0]
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

def generate_multiple_counterfactuals_for_sample(ppo_model: PPO, env: PPOEnv, dataset_path: str, 
                                                sample_index: int, num_counterfactuals: int = 5,
                                                save_path: Optional[str] = None, 
                                                max_steps_per_sample: int = Config.MAX_STEPS_PER_SAMPLE,
                                                use_mcts: bool = False, 
                                                mcts_simulations: int = Config.MCTS_SIMULATIONS, 
                                                verbose: int = 1) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """
    Generate multiple counterfactuals for a single sample.

    Args:
        ppo_model: Trained PPO model.
        env: Environment for counterfactual generation.
        dataset_path: Path to the dataset CSV file.
        sample_index: Index of the sample to process.
        num_counterfactuals: Number of counterfactuals to generate.
        save_path: Path to save the generated counterfactuals.
        max_steps_per_sample: Maximum steps per counterfactual.
        use_mcts: Whether to use MCTS for action selection.
        mcts_simulations: Number of MCTS simulations per step.
        verbose: Verbosity level for logging.

    Returns:
        Tuple of (counterfactuals, original_df, counterfactual_df).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_counterfactuals} counterfactuals for sample index {sample_index}...")
    
    dataset_utils = DatasetUtils(dataset_path, verbose)
    if sample_index < 0 or sample_index >= len(dataset_utils.dataset):
        raise ValueError(f"Sample index {sample_index} is out of range")
    
    env.use_random_sampling = False
    mcts = PPOMCTS(env=env, ppo_model=ppo_model, num_simulations=mcts_simulations) if use_mcts else None
    if use_mcts:
        logger.info(f"Initialized MCTS with {mcts_simulations} simulations per step")
    
    counterfactuals, original_samples, counterfactual_samples = [], [], []
    success_count, total_steps = 0, 0
    start_time = time.time()
    result_saver = ResultSaver(verbose)
    
    env.current_instance_idx = sample_index
    for cf_idx in tqdm(range(num_counterfactuals), desc=f"Generating counterfactuals for sample {sample_index}"):
        success, best_info, steps_taken = generate_single_counterfactual(
            env, ppo_model, mcts, max_steps_per_sample, Config.MAX_TRIES, use_mcts, verbose)
        
        original_features = env.original_features
        original_prediction = env.original_prediction
        counterfactuals.append({
            'sample_id': sample_index,
            'counterfactual_id': cf_idx,
            'data_index': sample_index,
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
    
    if counterfactuals:
        coverage, distance, implausibility, sparsity, actionability = get_metrics(
            original_df, counterfactual_df, counterfactuals, env.constraints, 
            dataset_utils.feature_columns, dataset_utils.dataset, verbose)
        diversity = get_diversity(counterfactual_df)
        logger.info(f"\nFinal statistics for sample {sample_index}:")
        logger.info(f"Coverage: {coverage:.2%} ({success_count}/{num_counterfactuals})")
        logger.info(f"Mean distance of counterfactuals: {distance:.2f}")
        logger.info(f"Mean sparsity: {sparsity}")
        logger.info(f"Mean implausibility: {implausibility:.2f}")
        logger.info(f"Mean actionability: {actionability:.2f}")
        logger.info(f"Diversity: {diversity:.2f}")
        logger.info(f"Average steps per successful counterfactual: {(total_steps / success_count if success_count > 0 else 0):.2f}")

    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")
    
    if save_path and counterfactuals:
        result_saver.save_counterfactuals(counterfactuals, dataset_utils.dataset, save_path, multiple=True)
        kpi_base_path = os.path.splitext(save_path)[0]
        original_df.to_csv(f"{kpi_base_path}_original.csv", index=False)
        counterfactual_df.to_csv(f"{kpi_base_path}_counterfactual.csv", index=False)
    
    env.use_random_sampling = True
    return counterfactuals, original_df, counterfactual_df

def train_grpo_for_counterfactuals(dataset_path: str, logs_dir: str, save_dir: str,
                                  total_iterations: int = Config.TOTAL_TIMESTEPS,
                                  steps_per_iteration: int = Config.GRPO_STEPS_PER_ITERATION,
                                  mu_iterations: int = Config.GRPO_MU_ITERATIONS,
                                  mode: str = 'new', constraints: Optional[Dict[str, str]] = None,
                                  verbose: int = 1) -> Tuple[GRPOTrainer, GRPOEnv]:
    """
    Train or load a GRPO model for counterfactual generation.

    Args:
        dataset_path: Path to the dataset CSV file.
        logs_dir: Directory for training logs.
        save_dir: Directory to save the model checkpoints.
        total_iterations: Number of outer iterations (I in Deepseek GRPO algorithm).
        steps_per_iteration: Number of inner steps per iteration (M in algorithm).
        mu_iterations: Number of GRPO inner iterations (μ in algorithm).
        mode: Training mode ('new', 'load', or 'continue').
        constraints: Dictionary of feature constraints.
        verbose: Verbosity level for logging.

    Returns:
        Tuple of (GRPOTrainer, GRPOEnv).
    """
    logger = logging.getLogger(__name__)
    valid_modes = ['new', 'load', 'continue']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    dataset_utils = DatasetUtils(dataset_path, verbose)
    classifier = load_classifier_model(dataset_path, None, dataset_utils.num_features,
                                      dataset_utils.num_classes, verbose)

    env = GRPOEnv(dataset_path=dataset_path, model=classifier)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = GRPOPolicy(obs_dim, action_dim)
    policy_model_path = os.path.join(save_dir, f"grpo_policy_{os.path.splitext(os.path.basename(dataset_path))[0]}.pt")

    if mode in ['load', 'continue'] and os.path.exists(policy_model_path):
        logger.info(f"Loading existing GRPO policy from {policy_model_path}")
        try:
            policy.load_state_dict(torch.load(policy_model_path))
            logger.info("Existing GRPO policy loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading existing model: {e}")
            if mode == 'load':
                raise
            logger.info("Falling back to training a new GRPO policy...")
            mode = 'new'

    trainer = GRPOTrainer(
        env=env,
        policy=policy,
        learning_rate=Config.LEARNING_RATE_NEW,
        group_size=Config.GRPO_GROUP_SIZE,
        clip_range=Config.CLIP_RANGE,
        kl_coef=Config.GRPO_KL_COEF,
        mu_iterations=mu_iterations
    )

    if mode in ['new', 'continue']:
        logger.info(f"Starting GRPO training for {total_iterations} iterations...")
        for iteration in range(total_iterations):
            trainer.update_reference_policy()
            logger.info(f"\n=== Iteration {iteration+1}/{total_iterations} ===")
            
            for step in range(steps_per_iteration):
                step_stats = trainer.train_step(num_episodes=5)
                
                if step % 1 == 0 and verbose > 0:
                    logger.info(f"  Step {step+1}/{steps_per_iteration}: "
                               f"Loss={step_stats['policy_loss']:.4f}, "
                               f"Success={step_stats['success_rate']:.2%}, "
                               f"Mean Reward={step_stats['mean_reward']:.2f}")

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
        logger.info(f"Final GRPO model saved: {final_path}")

    return trainer, env

def generate_counterfactuals_grpo(trainer: GRPOTrainer, env: GRPOEnv, dataset_path: str, 
                                 save_path: Optional[str] = None, specific_indices: Optional[List[int]] = None,
                                 max_steps_per_sample: int = Config.MAX_STEPS_PER_SAMPLE, 
                                 verbose: int = 1) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    logger = logging.getLogger(__name__)
    logger.info("Generating counterfactuals using GRPO...")
    
    dataset_utils = DatasetUtils(dataset_path, verbose)
    
    indices_to_use = specific_indices or list(range(len(dataset_utils.dataset)))
    logger.info(f"Processing {len(indices_to_use)} samples")
    
    counterfactuals, original_samples, counterfactual_samples = [], [], []
    success_count, total_steps = 0, 0
    start_time = time.time()
    result_saver = ResultSaver(verbose)
    
    trainer.policy.eval()
    
    for i, idx in tqdm(enumerate(indices_to_use), total=len(indices_to_use), desc="Generating counterfactuals"):
        env.current_instance_idx = idx
        obs = env.reset()
        
        done, steps_taken = False, 0
        while not done and steps_taken < max_steps_per_sample:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _ = trainer.policy.get_action_and_log_prob(obs_tensor, deterministic=True)
                action = action.item()
            
            obs, reward, done, info = env.step(action)
            steps_taken += 1
        
        original_features = env.original_features
        original_prediction = env.original_prediction
        success = info['counterfactual_found']
        
        counterfactuals.append({
            'sample_id': i,
            'data_index': idx,
            'original_features': original_features,
            'counterfactual_features': info['modified_features'] if success else None,
            'original_prediction': original_prediction,
            'counterfactual_prediction': info['modified_prediction'] if success else original_prediction,
            'distance': info['distance'] if success else float('inf'),
            'steps': steps_taken,
            'success': success
        })
        original_samples.append(original_features)
        counterfactual_samples.append(info['modified_features'] if success else original_features)
        success_count += success
        total_steps += steps_taken
    
    original_df = pd.DataFrame(original_samples, columns=dataset_utils.feature_columns)
    counterfactual_df = pd.DataFrame(counterfactual_samples, columns=dataset_utils.feature_columns)
    
    if counterfactuals:
        coverage, distance, implausibility, sparsity, actionability = get_metrics(
            original_df, counterfactual_df, counterfactuals, env.constraints, 
            dataset_utils.feature_columns, dataset_utils.dataset, verbose)
        diversity = get_diversity(counterfactual_df)
        logger.info(f"\nFinal GRPO statistics:")
        logger.info(f"Coverage: {coverage:.2%} ({success_count}/{len(indices_to_use)})")
        logger.info(f"Mean distance: {distance:.2f}")
        logger.info(f"Mean sparsity: {sparsity}")
        logger.info(f"Mean implausibility: {implausibility:.2f}")
        logger.info(f"Mean actionability: {actionability:.2f}")
        logger.info(f"Diversity: {diversity:.2f}")
        logger.info(f"Average steps per successful counterfactual: {(total_steps / success_count if success_count > 0 else 0):.2f}")
    
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")
    
    if save_path and counterfactuals:
        result_saver.save_counterfactuals_to_csv(counterfactuals, original_df, counterfactual_df, save_path)
    
    return counterfactuals, original_df, counterfactual_df
def calculate_actionability(original_features: np.ndarray, counterfactual_features: np.ndarray, 
                           constraints: Dict[str, str], feature_columns: List[str]) -> int:
    """
    Calculate actionability score for a counterfactual based on constraints.

    Args:
        original_features: Original feature values.
        counterfactual_features: Counterfactual feature values.
        constraints: Dictionary of feature constraints.
        feature_columns: List of feature column names.

    Returns:
        Actionability score (1 if all fixed features unchanged, 0 otherwise).
    """
    if not constraints:
        return 1
    for feature, constraint in constraints.items():
        if constraint == "fixed":
            idx = feature_columns.index(feature)
            if original_features[idx] != counterfactual_features[idx]:
                return 0
    return 1

def calculate_distance(original_features: np.ndarray, modified_features: np.ndarray) -> float:
    """
    Calculate L1 (Manhattan) distance between original and modified features.

    Args:
        original_features: Original feature vector.
        modified_features: Modified feature vector.

    Returns:
        L1 distance between the feature vectors.
    """
    categorical_indices = [i for i, feature in enumerate(original_features) 
                          if isinstance(feature, (str, bool))]
    dist = 0
    for i, (o, m) in enumerate(zip(original_features, modified_features)):
        dist += float(o != m) if i in categorical_indices else abs(o - m)
    return dist

def get_metrics(original_df: pd.DataFrame, counterfactual_df: pd.DataFrame, counterfactuals: List[Dict],
                constraints: Dict[str, str], feature_columns: List[str], original_data: pd.DataFrame,
                verbose: int = 1) -> Tuple[float, float, float, float, float]:
    """
    Calculate KPIs for generated counterfactuals.

    Args:
        original_df: DataFrame of original samples.
        counterfactual_df: DataFrame of counterfactual samples.
        counterfactuals: List of counterfactual dictionaries.
        constraints: Dictionary of feature constraints.
        feature_columns: List of feature column names.
        original_data: Original dataset for implausibility calculation.
        verbose: Verbosity level for logging.

    Returns:
        Tuple of (coverage, distance, implausibility, sparsity, actionability).
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Calculating KPIs for {len(counterfactuals)} counterfactuals...")
    
    if not counterfactuals:
        logger.warning("No counterfactuals generated. Returning default KPIs.")
        return 0, 0, 0, 0, 0
    
    # Calculate coverage
    success_count = sum(cf['success'] for cf in counterfactuals)
    num_samples = len(counterfactuals)
    coverage = success_count / num_samples if num_samples > 0 else 0
    logger.debug(f"Coverage: {coverage:.2%} ({success_count}/{num_samples})")
    
    # Calculate distance
    distance = np.mean([cf['distance'] for cf in counterfactuals]) if counterfactuals else 0
    logger.debug(f"Mean distance of counterfactuals: {distance:.2f}")
    
    # Calculate implausibility
    implausibility_scores = []
    for cf in counterfactuals:
        if cf['counterfactual_features'] is not None:
            cfe = np.concatenate((cf['counterfactual_features'], [cf['counterfactual_prediction']]))
            closest_sample = get_closest_samples(cfe, original_data, X=5, require_different_outcome=False).iloc[0]
            implausibility_scores.append(calculate_distance(cfe[:-1], closest_sample[:-1]))
    implausibility = np.mean(implausibility_scores) if implausibility_scores else 0
    logger.debug(f"Mean implausibility of counterfactuals: {implausibility:.2f}")
    
    # Calculate sparsity
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
    
    # Calculate actionability
    actionability_scores = [
        calculate_actionability(cf['original_features'], cf['counterfactual_features'], 
                              constraints, feature_columns)
        for cf in counterfactuals if cf['counterfactual_features'] is not None
    ]
    actionability = np.mean(actionability_scores) if actionability_scores else 0
    logger.debug(f"Mean actionability of counterfactuals: {actionability:.2f}")
    
    return coverage, distance, implausibility, sparsity, actionability

def get_diversity(counterfactual_df: pd.DataFrame):
    """
    Calculate diversity of counterfactuals for a unique sample. 
    Diversity is the smallest distance of a counterfactual with his furthest neighbor.

    Args:
        counterfactual_df: DataFrame of counterfactual samples.

    Returns:
        Diversity score (Higher value means better diversity).
    """
    logger = logging.getLogger(__name__)
    if counterfactual_df.empty:
        return 0.0
    
    diversities = []
    num_counterfactuals = len(counterfactual_df)

    # For each counterfactual, calculate the distance to all others and store the maximum distance
    for i in range(num_counterfactuals):
        distances = []
        for j in range(num_counterfactuals):
            if i != j:
                dist = calculate_distance(counterfactual_df.iloc[i].values[:-1], 
                                          counterfactual_df.iloc[j].values[:-1])
                distances.append(dist)
        if distances:
            diversities.append(min(distances))
    logger.debug(diversities)
    diversity = round(min(diversities),2) if diversities else 0.0

    return diversity

def main():
    """Main function to train or load PPO model and generate counterfactuals."""
    logger = logging.getLogger(__name__)
    setup_directories([Config.DATA_DIR, Config.LOGS_DIR, Config.SAVE_DIR])
    
    dataset_path = Config.DATASET_PATH
    constraints = Config.CONSTRAINTS
    indices_to_use = Config.INDICES_TO_USE
    
    logger.info("Processing PPO model...")
    ppo_model, env = train_ppo_for_counterfactuals(
        dataset_path=dataset_path,
        logs_dir=Config.LOGS_DIR,
        save_dir=Config.SAVE_DIR,
        total_timesteps=Config.TOTAL_TIMESTEPS,
        mode=Config.TRAINING_MODE,
        constraints=constraints,
        verbose=1
    )
    
    if ppo_model is not None:
        logger.info("Generating counterfactuals using standard PPO...")
        # generate_counterfactuals(
        #     ppo_model=ppo_model,
        #     env=env,
        #     dataset_path=dataset_path,
        #     save_path=os.path.join(Config.DATA_DIR, 
        #                           f"generated_counterfactuals_ppo_{os.path.splitext(os.path.basename(dataset_path))[0]}.csv"),
        #     max_steps_per_sample=Config.MAX_STEPS_PER_SAMPLE,
        #     use_mcts=False,
        #     mcts_simulations=Config.MCTS_SIMULATIONS,
        #     specific_indices=indices_to_use,
        #     verbose=1
        # )

        logger.info("Generating multiple counterfactuals for a specific sample...")
        sample_index = 0  # Change this to the desired sample index
        _, _, counterfactual_df = generate_multiple_counterfactuals_for_sample(
            ppo_model=ppo_model,
            env=env,
            dataset_path=dataset_path,
            sample_index=sample_index,
            num_counterfactuals=10,
            save_path=os.path.join(Config.DATA_DIR, 
                                  f"generated_counterfactuals_sample_{sample_index}.csv"),
            max_steps_per_sample=Config.MAX_STEPS_PER_SAMPLE,
            use_mcts=False,
            verbose=1
        )

    # logger.info("Processing GRPO model...")
    # grpo_trainer, grpo_env = train_grpo_for_counterfactuals(
    #     dataset_path=dataset_path,
    #     logs_dir=Config.LOGS_DIR,
    #     save_dir=Config.SAVE_DIR,
    #     total_iterations=Config.GRPO_TOTAL_ITERATIONS,
    #     steps_per_iteration=Config.GRPO_STEPS_PER_ITERATION,
    #     mu_iterations=Config.GRPO_MU_ITERATIONS,
    #     mode=Config.TRAINING_MODE,
    #     constraints=constraints,
    #     verbose=1
    # )
    
    # logger.info("Generating counterfactuals using GRPO...")
    # generate_counterfactuals_grpo(
    #     trainer=grpo_trainer,
    #     env=grpo_env,
    #     dataset_path=dataset_path,
    #     save_path=os.path.join(Config.DATA_DIR, 
    #                             f"generated_counterfactuals_grpo_{os.path.splitext(os.path.basename(dataset_path))[0]}.csv"),
    #     specific_indices=indices_to_use,
    #     verbose=1
    # )

if __name__ == "__main__":
    main()