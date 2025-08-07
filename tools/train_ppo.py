import logging
import os
from mcp.server.fastmcp import FastMCP
from PPO_train import train_ppo_for_counterfactuals, Config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

mcp = FastMCP("PPO Counterfactual Service")

@mcp.tool()
def train_ppo_tool(
    dataset_path: str,
    total_timesteps: int = Config.TOTAL_TIMESTEPS,
    training_mode: str = Config.TRAINING_MODE,
    constraints: dict = None
) -> dict:
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Training/loading PPO model for dataset: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset not found at {dataset_path}")
        ppo_model, env = train_ppo_for_counterfactuals(
            dataset_path=dataset_path,
            total_timesteps=total_timesteps,
            mode=training_mode,
            constraints=constraints or {},
            verbose=1
        )
        model_path = os.path.join(Config.SAVE_DIR, f"{Config.CHECKPOINT_PREFIX}_{os.path.splitext(os.path.basename(dataset_path))[0]}_final.zip")
        logger.info(f"PPO model saved/loaded at: {model_path}")
        return {
            "status": "success",
            "model_path": model_path,
            "message": f"PPO model {'trained' if training_mode in ['new', 'continue'] else 'loaded'} successfully"
        }
    except Exception as e:
        logger.error(f"Error training/loading PPO model: {e}")
        return {"error": str(e)}