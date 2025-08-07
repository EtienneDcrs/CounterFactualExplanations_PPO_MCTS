import os
import logging
import json
from typing import Optional
from fastmcp import FastMCP
from ppo_tools import train_ppo_model_tool, generate_counterfactuals_tool, calculate_counterfactual_metrics_tool

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

try:
    mcp = FastMCP("Counterfactual PPO Server", host="0.0.0.0", port=8080)
except Exception as e:
    logging.error(f"Failed to initialize FastMCP: {e}")
    raise

@mcp.tool()
def train_ppo_model(dataset_path: str, model_path: Optional[str] =None, total_timesteps: int = 400000, training_mode: str = 'new') -> dict:
    """
    Trains a PPO model for counterfactual generation.

    Args:
        dataset_path (str): Path to the CSV dataset.
        total_timesteps (int): Total timesteps for PPO training (default: 400000).
        training_mode (str): Training mode ('new', 'load', or 'continue').

    Returns:
        dict: {
            'status': str (success or error),
            'model_path': str (path where the model is saved),
            'message': str (details about the training process or error)
        }
    """
    try:
        result = train_ppo_model_tool(dataset_path, model_path,total_timesteps, training_mode)
        return result
    except Exception as e:
        logging.error(f"Error during PPO training: {e}")
        return {
            "status": "error",
            "model_path": "",
            "message": f"Failed to train PPO model: {str(e)}"
        }

@mcp.tool()
def generate_counterfactuals(dataset_path: str, model_path: str, max_steps_per_sample: int = 250, use_mcts: bool = False, mcts_simulations: int = 15, specific_indices: list = None) -> dict:
    """
    Generates counterfactuals for a dataset using a trained PPO model.

    Args:
        dataset_path (str): Path to the CSV dataset.
        model_path (str): Path to the trained PPO model.
        max_steps_per_sample (int): Maximum steps per sample for counterfactual generation (default: 250).
        use_mcts (bool): Whether to use Monte Carlo Tree Search (default: False).
        mcts_simulations (int): Number of MCTS simulations (default: 15).
        specific_indices (list): List of specific sample indices to process (default: None, processes all).

    Returns:
        dict: {
            'status': str (success or error),
            'counterfactual_csv_path': str (path where counterfactuals are saved),
            'message': str (details about the generation process or error)
        }
    """
    try:
        result = generate_counterfactuals_tool(dataset_path, model_path, max_steps_per_sample, use_mcts, mcts_simulations, specific_indices)
        return result
    except Exception as e:
        logging.error(f"Error during counterfactual generation: {e}")
        return {
            "status": "error",
            "counterfactual_csv_path": "",
            "message": f"Failed to generate counterfactuals: {str(e)}"
        }

@mcp.tool()
def calculate_counterfactual_metrics(original_csv_path: str, counterfactual_csv_path: str, constraints: dict = None) -> dict:
    """
    Calculates KPIs for counterfactuals based on original and counterfactual CSV files.

    Args:
        original_csv_path (str): Path to CSV file containing original feature samples (with 'sample_id' column).
        counterfactual_csv_path (str): Path to CSV file containing counterfactual feature samples (with 'sample_id' and 'counterfactual_found' columns).
        constraints (dict): Dictionary of feature constraints (e.g., {"feature_name": "fixed"}).

    Returns:
        dict: {
            'status': str (success or error),
            'metrics': {
                'coverage': float,
                'distance': float,
                'implausibility': float,
                'sparsity': float,
                'actionability': float,
                'diversity': float
            },
            'message': str (details about the metrics calculation or error)
        }
    """
    try:
        constraints = constraints or {}
        result = calculate_counterfactual_metrics_tool(original_csv_path, counterfactual_csv_path, constraints)
        return result
    except Exception as e:
        logging.error(f"Error during metrics calculation: {e}")
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

@mcp.resource("counterfactual://guidelines/ppo")
def get_ppo_guidelines() -> str:
    """Guidelines for using PPO for counterfactual generation."""
    guidelines = {
        "ppo_training": {
            "description": "Proximal Policy Optimization (PPO) is used to train a model for generating counterfactual explanations.",
            "parameters": {
                "dataset_path": "Path to the CSV dataset with features and target column.",
                "total_timesteps": "Total number of training iterations (e.g., 400000).",
                "training_mode": "Options: 'new' (train from scratch), 'load' (load existing model), 'continue' (continue training)."
            },
            "best_practices": [
                "Ensure dataset has no missing values.",
                "Use a sufficient number of timesteps for convergence.",
                "Validate model performance using evaluation metrics."
            ]
        },
        "counterfactual_generation": {
            "description": "Generates counterfactuals to explain model predictions by finding alternative feature values that change the outcome.",
            "parameters": {
                "dataset_path": "Path to the CSV dataset.",
                "model_path": "Path to the trained PPO model.",
                "max_steps_per_sample": "Maximum steps to find a counterfactual per sample.",
                "use_mcts": "Whether to use Monte Carlo Tree Search for exploration.",
                "mcts_simulations": "Number of MCTS simulations."
            }
        },
        "metrics": {
            "description": "Key performance indicators (KPIs) for evaluating counterfactuals.",
            "kpis": {
                "coverage": "Fraction of samples for which counterfactuals were found.",
                "distance": "Mean L1 distance between original and counterfactual features.",
                "implausibility": "Mean distance to the closest real sample in the dataset.",
                "sparsity": "Average number of features changed in counterfactuals.",
                "actionability": "Fraction of counterfactuals respecting feature constraints.",
                "diversity": "Minimum distance between counterfactuals for diversity."
            }
        }
    }
    return json.dumps(guidelines, indent=2)

@mcp.resource("counterfactual://model/info")
def get_model_information() -> str:
    """Information about the PPO counterfactual model."""
    model_info = {
        "model_type": "Proximal Policy Optimization (PPO)",
        "input_features": "Dataset features (numerical and categorical, defined by the provided CSV).",
        "output": {
            "counterfactuals": "Alternative feature values that change the model's prediction.",
            "metrics": "KPIs including coverage, distance, implausibility, sparsity, actionability, and diversity."
        },
        "usage_notes": [
            "Model trained on provided dataset for counterfactual generation.",
            "Counterfactuals aim to provide actionable insights for model explanations.",
            "Always validate counterfactuals against domain knowledge."
        ],
        "limitations": [
            "Dependent on dataset quality and feature representation.",
            "May require tuning for optimal performance.",
            "MCTS increases computation time but may improve results."
        ]
    }
    return json.dumps(model_info, indent=2)

if __name__ == "__main__":
    try:
        logging.info("Launching MCP server with Counterfactual PPO Tools...")
        mcp.run("streamable-http")
    except Exception as e:
        logging.error(f"Failed to run MCP server: {e}")
        raise