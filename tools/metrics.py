import logging
import pandas as pd
from typing import List, Dict
from fastmcp import FastMCP
from PPO_train import get_metrics, get_diversity, DatasetUtils

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Initialize FastMCP
mcp = FastMCP("Metrics Calculator", host="0.0.0.0", port=8080)

@mcp.tool()
def calculate_metrics_tool(
    dataset_path: str,
    counterfactuals: List[Dict],
    original_data: List[Dict],
    counterfactual_data: List[Dict],
    constraints: Dict[str, str] = None
) -> dict:
    """
    Calculates metrics for generated counterfactuals, including coverage, distance, implausibility, sparsity, and actionability.
    
    Args:
        dataset_path: Path to the dataset CSV file.
        counterfactuals: List of counterfactual dictionaries.
        original_data: List of dictionaries representing original samples.
        counterfactual_data: List of dictionaries representing counterfactual samples.
        constraints: Dictionary of feature constraints.
    
    Returns:
        Dictionary containing calculated metrics.
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Calculating metrics for counterfactuals")
        
        # Convert input lists to DataFrames
        original_df = pd.DataFrame(original_data)
        counterfactual_df = pd.DataFrame(counterfactual_data)
        dataset_utils = DatasetUtils(dataset_path, verbose=1)
        
        # Calculate metrics
        coverage, distance, implausibility, sparsity, actionability = get_metrics(
            original_df=original_df,
            counterfactual_df=counterfactual_df,
            counterfactuals=counterfactuals,
            constraints=constraints or {},
            feature_columns=dataset_utils.feature_columns,
            original_data=dataset_utils.dataset,
            verbose=1
        )
        
        # Calculate diversity if applicable
        diversity = get_diversity(counterfactual_df) if len(counterfactual_df) > 1 else 0.0
        
        result = {
            "coverage": coverage,
            "mean_distance": distance,
            "mean_implausibility": implausibility,
            "mean_sparsity": sparsity,
            "mean_actionability": actionability,
            "diversity": diversity
        }
        logger.info("Metrics calculation completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        logging.info("Launching Metrics Calculator MCP server...")
        mcp.run("streamable-http")
    except Exception as e:
        logging.error(f"Failed to run MCP server: {e}")
        raise