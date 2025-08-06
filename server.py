import logging
import json
from fastmcp import FastMCP
import tools.train_ppo  # Import to trigger @mcp.tool() registration
import tools.counterfactuals
import tools.metrics

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Initialize FastMCP server
try:
    mcp = FastMCP("PPO Counterfactual Service", host="0.0.0.0", port=8080)
except Exception as e:
    logging.error(f"Failed to initialize FastMCP: {e}")
    raise

# Resource endpoint for dataset information
@mcp.resource("counterfactual://dataset/info")
def get_dataset_info() -> str:
    """Returns information about the dataset and counterfactual generation."""
    info = {
        "dataset_requirements": {
            "format": "CSV",
            "features": "Numerical and categorical features supported",
            "target": "Last column should be the target variable"
        },
        "counterfactual_modes": ["all_samples", "single_sample"],
        "constraints": ["fixed", "increase", "decrease", "change"]
    }
    return json.dumps(info, indent=2)

# Resource endpoint for model information
@mcp.resource("counterfactual://model/info")
def get_model_info() -> str:
    """Returns information about the PPO model and counterfactual generation."""
    info = {
        "model_type": "Proximal Policy Optimization (PPO) with Stable Baselines3",
        "environment": "Custom PPOEnv for counterfactual generation",
        "constraints_supported": ["fixed", "increase", "decrease", "change"],
        "metrics": ["coverage", "mean_distance", "mean_implausibility", "mean_sparsity", "mean_actionability", "diversity"]
    }
    return json.dumps(info, indent=2)

if __name__ == "__main__":
    try:
        logging.info("Launching PPO Counterfactual MCP server...")
        mcp.run(transport="streamable-http")  # Use 'streamable-http' for FastMCP 2.2.0
    except Exception as e:
        logging.error(f"Failed to run MCP server: {e}")
        raise