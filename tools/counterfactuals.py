import logging
import pandas as pd
import os
from typing import Optional, Dict, List
from fastmcp import FastMCP
from PPO_train import train_ppo_for_counterfactuals, generate_counterfactuals, generate_multiple_counterfactuals_for_sample, Config, DatasetUtils

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Initialize FastMCP
mcp = FastMCP("PPO Counterfactual Service", host="0.0.0.0", port=8080)

@mcp.tool()
def generate_counterfactuals_tool(
    dataset_path: str,
    specific_indices: Optional[List[int]] = None,
    constraints: Optional[Dict[str, str]] = None,
    max_steps_per_sample: int = 250,
    use_mcts: bool = False,
    mcts_simulations: int = 15
) -> dict:
    """
    Generates counterfactual explanations for all samples or specific indices in a dataset.
    
    Args:
        dataset_path: Path to the dataset CSV file.
        specific_indices: List of indices to process (optional).
        constraints: Dictionary of feature constraints.
        max_steps_per_sample: Maximum steps per counterfactual.
        use_mcts: Whether to use Monte Carlo Tree Search.
        mcts_simulations: Number of MCTS simulations.
    
    Returns:
        Dictionary containing counterfactuals, original DataFrame, and counterfactual DataFrame.
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Generating counterfactuals for dataset: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset not found at {dataset_path}")
        
        ppo_model, env = train_ppo_for_counterfactuals(
            dataset_path=dataset_path,
            total_timesteps=Config.TOTAL_TIMESTEPS,
            mode=Config.TRAINING_MODE,
            constraints=constraints or {},
            verbose=1
        )
        
        counterfactuals, original_df, counterfactual_df = generate_counterfactuals(
            ppo_model=ppo_model,
            env=env,
            dataset_path=dataset_path,
            save_path=None,
            specific_indices=specific_indices,
            max_steps_per_sample=max_steps_per_sample,
            use_mcts=use_mcts,
            mcts_simulations=mcts_simulations,
            verbose=1
        )
        
        result = {
            "counterfactuals": counterfactuals,
            "original_data": original_df.to_dict(orient="records"),
            "counterfactual_data": counterfactual_df.to_dict(orient="records")
        }
        logger.info("Counterfactual generation completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error generating counterfactuals: {e}")
        return {"error": str(e)}

@mcp.tool()
def generate_multiple_counterfactuals_tool(
    dataset_path: str,
    sample_index: int,
    num_counterfactuals: int = 5,
    constraints: Optional[Dict[str, str]] = None,
    max_steps_per_sample: int = 250,
    use_mcts: bool = False,
    mcts_simulations: int = 15
) -> dict:
    """
    Generates multiple counterfactual explanations for a single sample.
    
    Args:
        dataset_path: Path to the dataset CSV file.
        sample_index: Index of the sample to process.
        num_counterfactuals: Number of counterfactuals to generate.
        constraints: Dictionary of feature constraints.
        max_steps_per_sample: Maximum steps per counterfactual.
        use_mcts: Whether to use Monte Carlo Tree Search.
        mcts_simulations: Number of MCTS simulations.
    
    Returns:
        Dictionary containing counterfactuals, original DataFrame, and counterfactual DataFrame.
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Generating {num_counterfactuals} counterfactuals for sample {sample_index}")
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset not found at {dataset_path}")
        
        dataset_utils = DatasetUtils(dataset_path, verbose=1)
        if sample_index < 0 or sample_index >= len(dataset_utils.dataset):
            raise ValueError(f"Invalid sample index: {sample_index}")
        
        ppo_model, env = train_ppo_for_counterfactuals(
            dataset_path=dataset_path,
            total_timesteps=Config.TOTAL_TIMESTEPS,
            mode=Config.TRAINING_MODE,
            constraints=constraints or {},
            verbose=1
        )
        
        counterfactuals, original_df, counterfactual_df = generate_multiple_counterfactuals_for_sample(
            ppo_model=ppo_model,
            env=env,
            dataset_path=dataset_path,
            sample_index=sample_index,
            num_counterfactuals=num_counterfactuals,
            save_path=None,
            max_steps_per_sample=max_steps_per_sample,
            use_mcts=use_mcts,
            mcts_simulations=mcts_simulations,
            verbose=1
        )
        
        result = {
            "counterfactuals": counterfactuals,
            "original_data": original_df.to_dict(orient="records"),
            "counterfactual_data": counterfactual_df.to_dict(orient="records")
        }
        logger.info("Counterfactual generation completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error generating multiple counterfactuals: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        logging.info("Launching Counterfactual Generator MCP server...")
        mcp.run(transport="streamable-http")
    except Exception as e:
        logging.error(f"Failed to run MCP server: {e}")
        raise