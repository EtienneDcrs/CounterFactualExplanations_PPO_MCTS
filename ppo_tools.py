import os
import pickle
import time
import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from PPO_env import PPOEnv, PPOMonitorCallback
from PPO_train import train_ppo_for_counterfactuals, generate_counterfactuals
from utils import get_metrics,get_categorical_columns, calculate_distance, get_closest_samples, calculate_actionability, get_diversity
from Classifier_model import Classifier, train_model

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ppo_training.log")]
)
logger = logging.getLogger(__name__)

# Increase recursion limit temporarily for debugging
sys.setrecursionlimit(2000)  # Default is 1000, increase cautiously

class Config:
    """Configuration constants for PPO training and counterfactual generation."""
    SAVE_DIR: str = 'ppo_models'
    DATA_DIR: str = 'data'
    LOGS_DIR: str = 'ppo_logs'
    CHECKPOINT_PREFIX: str = 'ppo_certifai'  # Aligned with PPO_train.py
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
    MAX_STEPS_PER_SAMPLE: int = 250
    CONSTRAINTS: Dict[str, str] = {}
    MCTS_SIMULATIONS: int = 15  # Added from PPO_train.py
    INDICES_TO_USE: Optional[List[int]] = list(range(100))  # Added from PPO_train.py

def setup_directories(directories: List[str]) -> None:
    """Create directories if they don't exist."""
    for directory in directories:
        logger.debug(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

def validate_dataset(dataset: pd.DataFrame) -> None:
    """Validate dataset for missing values and non-numeric features."""
    logger.debug(f"Validating dataset with shape: {dataset.shape}")
    if dataset.isnull().any().any():
        missing_count = dataset.isnull().sum().sum()
        logger.warning(f"Found {missing_count} missing values in dataset")
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns
    non_numeric_columns = [col for col in dataset.columns if col not in numeric_columns]
    if non_numeric_columns:
        logger.warning(f"Non-numeric columns detected: {non_numeric_columns}")

def train_ppo_model_tool(dataset_path: str,model_path:Optional[str] = None, output_ppo_model_path:Optional[str]=None, total_timesteps: int = 400000, training_mode: str = 'new') -> dict:
    """
    Train a PPO model for counterfactual generation using train_ppo_for_counterfactuals from PPO_train.py.

    Args:
        dataset_path: Path to the CSV dataset.
        output_ppo_model_path: Path to save the trained PPO model.
        total_timesteps: Total timesteps for training.
        training_mode: 'new', 'load', or 'continue'.

    Returns:
        dict: Training result with status, model path, and message.
    """
    logger.info(f"Starting PPO training: dataset={dataset_path},\nmodel_path={model_path}, \noutput_ppo_model_path={output_ppo_model_path},\ntimesteps={total_timesteps}, \nmode={training_mode}")
    if type(total_timesteps) is not int or total_timesteps <= 0:
        logger.error("Invalid total_timesteps: must be a positive integer, using default 400000")
        total_timesteps = 400000  # Default value if invalid
        
    try:
        model,env = train_ppo_for_counterfactuals(dataset_path, model_path, output_ppo_model_path,
                                  logs_dir=Config.LOGS_DIR, save_dir= Config.SAVE_DIR,
                                  total_timesteps= total_timesteps, 
                                  mode = training_mode, 
                                  constraints = None,
                                  verbose= 1) 
        if model is None or env is None:
            raise ValueError("Model or environment initialization failed during training")
        else:
            logger.info("PPO model trained successfully")
            return {
                "status": "success",
                "model_path": model_path or os.path.join(Config.SAVE_DIR, f"ppo_model_{int(time.time())}.zip"),
                "message": f"PPO model trained successfully and saved to {model_path or 'default path'}"
            }
    except Exception as e:
        logger.error(f"Error in train_ppo_model_tool: {e}", exc_info=True)
        return {
            "status": "error",
            "model_path": "",
            "message": f"Failed to train PPO model: {str(e)}"
        }

def generate_counterfactuals_tool(
    dataset_path: str,
    ppo_path: str,
    classifier_path: str,
    max_steps_per_sample: int = 250, 
    use_mcts: bool = False, 
    mcts_simulations: int = 15, 
    specific_indices: Optional[List[int]] = None
) -> dict:
    """
    Generate counterfactuals using a trained PPO model, leveraging generate_counterfactuals from PPO_train.py.

    Args:
        dataset_path: Path to the CSV dataset.
        ppo_path: Path to the trained PPO model.
        classifier_path: Path to the trained classifier model (pickle file).
        max_steps_per_sample: Maximum steps per sample.
        use_mcts: Whether to use Monte Carlo Tree Search.
        mcts_simulations: Number of MCTS simulations.
        specific_indices: List of specific sample indices to process.

    Returns:
        dict: Generation result with status, counterfactual CSV path, and message.
    """
    logger.info(f"Starting counterfactual generation: dataset={dataset_path}, model={ppo_path}, "
                f"classifier={classifier_path}, use_mcts={use_mcts}, mcts_simulations={mcts_simulations}")
    try:
        # Setup directories
        setup_directories([Config.DATA_DIR])

        # Load and validate dataset
        logger.debug(f"Loading dataset from {dataset_path}")
        dataset = pd.read_csv(dataset_path)
        validate_dataset(dataset)
        logger.debug(f"Dataset validated: shape={dataset.shape}")

        # Load classifier model (required by PPOEnv)
        if not os.path.exists(classifier_path):
            logger.debug("Training new classifier model for PPOEnv")
            classifier = train_model(dataset_path)  # From Classifier_model.py
            with open(classifier_path, 'wb') as f:
                pickle.dump(classifier, f)
        else:
            logger.debug(f"Loading classifier model from {classifier_path}")
            with open(classifier_path, 'rb') as f:
                classifier = pickle.load(f)
        logger.debug("Classifier model loaded successfully")

        # Initialize environment
        logger.debug("Initializing PPOEnv for counterfactual generation")
        env = PPOEnv(
            dataset_path=dataset_path,
            model=classifier,  # Use classifier model
            label_encoders=None,
            scaler=None,
            constraints=Config.CONSTRAINTS,
            use_random_sampling=True,
            max_steps=max_steps_per_sample,
            verbose=1
        )
        logger.debug("PPOEnv initialized successfully")

        # Load PPO model
        logger.debug(f"Loading PPO model from {ppo_path}")
        if not os.path.exists(ppo_path):
            logger.error(f"PPO model not found at {ppo_path}")
            raise FileNotFoundError(f"PPO model not found at {ppo_path}")
        model = PPO.load(ppo_path, env=env)
        logger.debug("PPO model loaded successfully")

        # Generate save path
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        save_path = os.path.join(Config.DATA_DIR, 
                                f"generated_counterfactuals_{'mcts' if use_mcts else 'ppo'}_{dataset_name}.csv")

        # Call generate_counterfactuals from PPO_train.py
        logger.debug("Calling generate_counterfactuals from PPO_train.py")
        generate_counterfactuals(
            ppo_model=model,
            env=env,
            dataset_path=dataset_path,
            save_path=save_path,
            max_steps_per_sample=max_steps_per_sample,
            use_mcts=use_mcts,
            mcts_simulations=mcts_simulations,
            specific_indices=specific_indices,
            verbose=1
        )

        logger.info(f"Counterfactuals saved to {save_path}")
        return {
            "status": "success",
            "counterfactual_csv_path": save_path,
            "message": f"Counterfactuals generated and saved to {save_path}"
        }
    except Exception as e:
        logger.error(f"Error in generate_counterfactuals: {e}", exc_info=True)
        return {
            "status": "error",
            "counterfactual_csv_path": "",
            "message": f"Failed to generate counterfactuals: {str(e)}"
        }

def calculate_counterfactual_metrics_tool(original_csv_path: str, counterfactual_csv_path: str, 
                                    constraints: Optional[Dict[str, str]] = None) -> dict:
    """
    Calculate KPIs for counterfactuals based on CSV files.

    Args:
        original_csv_path: Path to original dataset CSV.
        counterfactual_csv_path: Path to counterfactual CSV.
        constraints: Dictionary of feature constraints.

    Returns:
        dict: Metrics result with status, KPIs, and message.
    """
    logger.info(f"Calculating metrics: original={original_csv_path}, counterfactual={counterfactual_csv_path}")
    try:
        constraints = constraints or Config.CONSTRAINTS
        coverage, distance, implausibility, sparsity, actionability, diversity = get_metrics(
            original_csv_path, counterfactual_csv_path, constraints
        )
        logger.info(f"Metrics calculated: coverage={coverage:.2%}, distance={distance:.2f}, "
                    f"implausibility={implausibility:.2f}, sparsity={sparsity:.2f}, "
                    f"actionability={actionability:.2f}, diversity={diversity:.2f}")
        return {
            "status": "success",
            "metrics": {
                "coverage": round(float(coverage), 4),
                "distance": round(float(distance), 4),
                "implausibility": round(float(implausibility), 4),
                "sparsity": round(float(sparsity), 4),
                "actionability": round(float(actionability), 4),
                "diversity": round(float(diversity), 4)
            },
            "message": "Metrics calculated successfully"
        }
    except Exception as e:
        logger.error(f"Error in calculate_counterfactual_metrics: {e}", exc_info=True)
        return {
            "status": "error",
            "metrics": {
                "coverage": 0.0,
                "distance": 0.0,
                "implausibility": 0.0,
                "sparsity": 0.0,
                "actionability": 0.0,
                "diversity": 0.0
            },
            "message": f"Failed to calculate metrics: {str(e)}"
        }
