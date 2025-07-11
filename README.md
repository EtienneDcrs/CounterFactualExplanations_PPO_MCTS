# Counterfactual Explanations Generation Algorithm

## Overview
This project implements a Proximal Policy Optimization (PPO) based algorithm for generating counterfactual explanations for classification models. Counterfactual explanations identify minimal changes to input features that result in a different prediction, helping to interpret and understand machine learning model decisions.

The algorithm uses reinforcement learning (RL) to generate counterfactuals, with support for feature constraints, Monte Carlo Tree Search (MCTS) for action selection, and comprehensive evaluation metrics. It is designed to work with tabular datasets and includes utilities for dataset handling, model training, and result analysis.

## Features
- **PPO-based Counterfactual Generation**: Utilizes the Stable Baselines3 PPO implementation to train an RL agent for generating counterfactuals.
- **Custom Environment**: A Gym-compliant environment (`PPOEnv`) tailored for counterfactual generation with support for continuous and categorical features.
- **Feature Constraints**: Supports constraints like "increase", "decrease" (for numerical features), and "fixed" (for any feature).
- **MCTS Integration**: Optional Monte Carlo Tree Search for improved action selection during counterfactual generation.
- **Evaluation Metrics**: Calculates KPIs including coverage, distance, implausibility, sparsity, and actionability.
- **Flexible Dataset Handling**: Supports CSV datasets or NumPy arrays with automatic preprocessing for missing values.
- **Model Persistence**: Saves and loads trained PPO models, classifiers, label encoders, and scalers.
- **Logging and Monitoring**: Comprehensive logging and a custom callback (`PPOMonitorCallback`) for tracking training progress.

## Requirements
- Python 3.8+
- Libraries:
  - `torch`
  - `numpy`
  - `pandas`
  - `stable-baselines3`
  - `gym`
  - `tqdm`
  - `scikit-learn`

Install dependencies using:
```bash
pip install torch numpy pandas stable-baselines3 gym tqdm scikit-learn
```

## Project Structure
- **PPO_train.py**: Main script for training the PPO model and generating counterfactuals. Includes utilities for dataset handling, model initialization, training, and result saving.
- **PPO_env.py**: Defines the `PPOEnv` Gym environment, `DatasetHandler` for dataset preprocessing, and `PPOMonitorCallback` for training monitoring.
- **classification_models/**: Directory to store trained classifier models (`.pt` files) and associated encoders/scalers (`.pkl` files).
- **ppo_models/**: Directory to store trained PPO models (`.zip` files).
- **ppo_logs/**: Directory for TensorBoard logs.
- **data/**: Directory for input datasets and generated counterfactuals.

## Usage
1. **Prepare the Dataset**:
   - Place your dataset (CSV format) in the `data/` directory or provide a NumPy array.
   - Ensure the last column is the target variable.
   - Example dataset: `data/adult.csv`

2. **Configure the Algorithm**:
   - Modify the `Config` class in `PPO_train.py` to set parameters like:
     - `DATASET_NAME`: Name of the dataset (e.g., `adult`).
     - `TOTAL_TIMESTEPS`: Number of training timesteps (default: 75000).
     - `CONSTRAINTS`: Feature constraints (e.g., `{"age": "increase"}`).
     - `TRAINING_MODE`: `new`, `load`, or `continue` for training or loading models.
     - Other hyperparameters (e.g., `LEARNING_RATE_NEW`, `N_STEPS`, etc.).

3. **Run the Algorithm**:
   - Execute the main script to train the PPO model and generate counterfactuals:
     ```bash
     python PPO_train.py
     ```
   - The script will:
     - Load or train a classifier model.
     - Train or load a PPO model based on the `TRAINING_MODE`.
     - Generate counterfactuals for the specified indices (default: first 100 samples).
     - Save results to `data/generated_counterfactuals_ppo_<dataset_name>.csv`.

4. **Output**:
   - **Counterfactuals**: Saved as a CSV file with columns for sample ID, original and counterfactual features, predictions, distance, and steps taken.
   - **Original and Counterfactual DataFrames**: Saved as `*_original.csv` and `*_counterfactual.csv`.
   - **Metrics**: Logged metrics include:
     - **Coverage**: Proportion of samples with successful counterfactuals.
     - **Distance**: Mean L1 distance between original and counterfactual features.
     - **Implausibility**: Mean distance to the closest real sample.
     - **Sparsity**: Average number of changed features.
     - **Actionability**: Proportion of counterfactuals respecting "fixed" constraints.

5. **Optional MCTS**:
   - Enable MCTS for action selection by setting `use_mcts=True` in `generate_counterfactuals` or `generate_multiple_counterfactuals_for_sample`.
   - Adjust `MCTS_SIMULATIONS` in the `Config` class to control the number of simulations.

## Example
To generate counterfactuals for the Adult dataset:
```python
from PPO_train import train_ppo_for_counterfactuals, generate_counterfactuals

# Train or load PPO model
ppo_model, env = train_ppo_for_counterfactuals(
    dataset_path="data/adult.csv",
    mode="new",
    constraints={"age": "increase"}
)

# Generate counterfactuals
counterfactuals, original_df, counterfactual_df = generate_counterfactuals(
    ppo_model=ppo_model,
    env=env,
    dataset_path="data/adult.csv",
    save_path="data/generated_counterfactuals.csv",
    specific_indices=[0, 1, 2]
)
```

## Key Components
- **PPOEnv** (`PPO_env.py`):
  - A Gym environment for counterfactual generation.
  - Handles continuous and categorical features with appropriate actions.
  - Supports feature constraints and encodes features using provided scalers and label encoders.
  - Computes rewards based on confidence scores and counterfactual success.

- **DatasetHandler** (`PPO_env.py`):
  - Loads and preprocesses datasets, handling missing values and identifying feature types.

- **PPOMonitorCallback** (`PPO_env.py`):
  - Monitors training progress, logging steps, FPS, and success rates.

- **Config** (`PPO_train.py`):
  - Central configuration for hyperparameters and paths.

- **ResultSaver** (`PPO_train.py`):
  - Saves counterfactual results and associated DataFrames.

- **Metrics** (`PPO_train.py`):
  - Computes and logs evaluation metrics for generated counterfactuals.

## Notes
- The classifier model must be pre-trained or will be trained automatically if not found.
- The algorithm assumes the dataset is clean; missing values are dropped by default.
- For large datasets, consider using `batch_size` in `generate_counterfactuals` to save intermediate results.
- MCTS can improve counterfactual quality but increases computation time.

## Future Improvements
- Add support for additional distance metrics.
- Enhance MCTS with adaptive simulation counts.
- Implement parallel processing for faster counterfactual generation.
- Add visualization tools for counterfactual analysis.

## License
This project is licensed under the MIT License.
