import streamlit as st
import pandas as pd
import os
import logging
from logging import StreamHandler
from PPO_train import (
    train_ppo_for_counterfactuals,
    generate_counterfactuals,
    generate_multiple_counterfactuals_for_sample,
    get_metrics,
    get_diversity,
    Config,
    setup_directories,
    DatasetUtils
)
from PPO_env import DatasetHandler

# Custom Streamlit logging handler
class StreamlitHandler(StreamHandler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.logs = []

    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)
        self.container.markdown("\n".join(self.logs[-10:]))  # Show last 10 logs to avoid overflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app title
st.title("Counterfactual Explanation Generator")

# Initialize session state
if 'confirmed_dataset_path' not in st.session_state:
    st.session_state['confirmed_dataset_path'] = None
    st.session_state['confirmed_training_mode'] = Config.TRAINING_MODE
    st.session_state['total_timesteps'] = Config.TOTAL_TIMESTEPS
    st.session_state['specific_indices'] = None
    st.session_state['sample_index'] = 0
    st.session_state['num_counterfactuals'] = 5
    st.session_state['constraints'] = {}
    st.session_state['max_steps_per_sample'] = Config.MAX_STEPS_PER_SAMPLE
    st.session_state['use_mcts'] = False
    st.session_state['mcts_simulations'] = Config.MCTS_SIMULATIONS
    st.session_state['counterfactuals'] = None
    st.session_state['original_df'] = None
    st.session_state['counterfactual_df'] = None
    st.session_state['metrics'] = None
    st.session_state['diversity'] = None
    st.session_state['feature_columns'] = None
    st.session_state['categorical_columns'] = None
    st.session_state['temp_constraints'] = {}

# Sidebar menu for method selection
st.sidebar.header("Generation Method")
generation_mode = st.sidebar.selectbox(
    "Select Method",
    ["Counterfactuals for All Samples", "Multiple Counterfactuals for a Specific Sample"],
    key="generation_mode"
)

# Main section for inputs
st.header("Configure Parameters")

# Dataset Path or Upload
st.subheader("Specify Dataset")
dataset_option = st.radio("Dataset Source", ["Use Default Path", "Upload CSV File"], key="dataset_option")
if dataset_option == "Use Default Path":
    dataset_path_input = st.text_input("Dataset Path", Config.DATASET_PATH, key="dataset_path")
    if st.button("Load Dataset", key="load_dataset_path"):
        if dataset_path_input and os.path.exists(dataset_path_input):
            st.session_state['confirmed_dataset_path'] = dataset_path_input
            # Load dataset to get feature and categorical columns
            dataset_handler = DatasetHandler(dataset_path_input, verbose=0)
            st.session_state['feature_columns'] = dataset_handler.feature_order
            st.session_state['categorical_columns'] = dataset_handler.get_con_cat_columns()[1]
            st.success("Dataset loaded successfully!")
        else:
            st.error("Please provide a valid dataset path.")
else:
    uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"], key="upload_dataset")
    if uploaded_file and st.button("Load Uploaded Dataset", key="load_uploaded_dataset"):
        dataset_path = f"data/uploaded_{uploaded_file.name}"
        os.makedirs("data", exist_ok=True)
        with open(dataset_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['confirmed_dataset_path'] = dataset_path
        # Load dataset to get feature and categorical columns
        dataset_handler = DatasetHandler(dataset_path, verbose=0)
        st.session_state['feature_columns'] = dataset_handler.feature_order
        st.session_state['categorical_columns'] = dataset_handler.get_con_cat_columns()[1]
        st.success(f"Uploaded and loaded {uploaded_file.name}")

# Training Mode
st.subheader("Select Training Mode")
training_mode_input = st.selectbox(
    "Training Mode",
    ["new", "load", "continue"],
    index=["new", "load", "continue"].index(Config.TRAINING_MODE),
    key="training_mode_input"
)

# Total Timesteps (if applicable)
if training_mode_input in ["new", "continue"]:
    st.subheader("Specify Total Timesteps")
    total_timesteps = st.number_input(
        "Total Timesteps",
        min_value=1000,
        value=Config.TOTAL_TIMESTEPS,
        step=1000,
        key="total_timesteps"
    )
else:
    total_timesteps = st.session_state['total_timesteps']

# Specific Indices or Sample Index
if generation_mode == "Counterfactuals for All Samples":
    st.subheader("Specify Indices (Optional)")
    indices_input = st.text_input(
        "Specific Indices (comma-separated, e.g., 0,1,2 or leave blank for all)",
        "",
        key="specific_indices_input"
    )
else:
    st.subheader("Specify Sample Index")
    sample_index = st.number_input(
        "Sample Index",
        min_value=0,
        value=0,
        step=1,
        key="sample_index"
    )
    num_counterfactuals = st.number_input(
        "Number of Counterfactuals",
        min_value=1,
        value=5,
        step=1,
        key="num_counterfactuals"
    )

# Constraints
st.subheader("Specify Constraints")
if st.session_state['feature_columns']:
    # Display current constraints
    if st.session_state['temp_constraints']:
        st.write("Current Constraints:")
        for feature, constraint in st.session_state['temp_constraints'].items():
            st.write(f"{feature}: {constraint}")

    # Add new constraint
    available_features = [
        f for f in st.session_state['feature_columns']
        if f not in st.session_state['temp_constraints']
    ]
    if available_features:
        selected_feature = st.selectbox(
            "Select Feature to Constrain",
            available_features,
            key="constraint_feature"
        )
        # Determine available constraint types based on feature type
        if selected_feature in st.session_state['categorical_columns']:
            constraint_options = ["fixed", "change"]
        else:
            constraint_options = ["increase", "decrease", "fixed"]
        selected_constraint = st.selectbox(
            f"Constraint for {selected_feature}",
            constraint_options,
            key="constraint_type"
        )
        if st.button("Add Constraint", key="add_constraint"):
            st.session_state['temp_constraints'][selected_feature] = selected_constraint
            st.rerun()
else:
    st.write("Load a dataset to specify constraints.")

# Max Steps per Sample
st.subheader("Specify Max Steps per Sample")
max_steps_per_sample = st.number_input(
    "Max Steps per Sample",
    min_value=10,
    value=Config.MAX_STEPS_PER_SAMPLE,
    step=10,
    key="max_steps_per_sample"
)

# MCTS Options
st.subheader("Configure MCTS (Optional)")
use_mcts = st.checkbox("Use MCTS", value=False, key="use_mcts")
mcts_simulations = st.number_input(
    "MCTS Simulations",
    min_value=1,
    value=Config.MCTS_SIMULATIONS,
    step=1,
    key="mcts_simulations"
)

# Save Path
st.subheader("Save Path for Results")
save_path = st.text_input(
    "Save Path for Counterfactuals",
    os.path.join(Config.DATA_DIR, "generated_counterfactuals.csv"),
    key="save_path"
)

# Validate inputs and enable/disable Run button
run_button_disabled = False
if not st.session_state['confirmed_dataset_path'] or not os.path.exists(st.session_state['confirmed_dataset_path']):
    run_button_disabled = True
    st.error("Please load a valid dataset.")
elif generation_mode == "Multiple Counterfactuals for a Specific Sample":
    dataset_utils = DatasetUtils(st.session_state['confirmed_dataset_path'], verbose=0)
    if sample_index >= len(dataset_utils.dataset):
        run_button_disabled = True
        st.error(f"Sample index {sample_index} is out of range for the dataset.")

# Run Button
if st.button("Run", disabled=run_button_disabled, key="run_button"):
    # Create log container
    log_container = st.container()
    log_container.subheader("Training and Generation Logs")
    streamlit_handler = StreamlitHandler(log_container)
    logger.handlers = [streamlit_handler]  # Replace existing handlers to avoid duplicate logs
    logger.setLevel(logging.INFO)

    with st.spinner("Generating counterfactuals..."):
        try:
            # Use input values directly
            if generation_mode == "Counterfactuals for All Samples":
                if indices_input.strip():
                    try:
                        specific_indices = [int(i) for i in indices_input.split(",")]
                    except ValueError:
                        st.error("Invalid indices format. Please use comma-separated integers.")
                        st.stop()
                else:
                    specific_indices = None
            else:
                sample_index = sample_index
                num_counterfactuals = num_counterfactuals

            # Ensure directories exist
            setup_directories([Config.DATA_DIR, Config.LOGS_DIR, Config.SAVE_DIR])

            # Train or load PPO model
            logger.info("Starting PPO model training/loading...")
            ppo_model, env = train_ppo_for_counterfactuals(
                dataset_path=st.session_state['confirmed_dataset_path'],
                logs_dir=Config.LOGS_DIR,
                save_dir=Config.SAVE_DIR,
                total_timesteps=total_timesteps,
                mode=training_mode_input,
                constraints=st.session_state['temp_constraints'],
                verbose=1
            )
            logger.info("PPO model training/loading completed.")

            # Generate counterfactuals
            indices = list(range(100))
            logger.info("Starting counterfactual generation...")
            if generation_mode == "Counterfactuals for All Samples":
                counterfactuals, original_df, counterfactual_df = generate_counterfactuals(
                    ppo_model=ppo_model,
                    env=env,
                    dataset_path=st.session_state['confirmed_dataset_path'],
                    save_path=save_path,
                    specific_indices=indices,
                    max_steps_per_sample=max_steps_per_sample,
                    use_mcts=use_mcts,
                    mcts_simulations=mcts_simulations,
                    verbose=1
                )
            else:
                counterfactuals, original_df, counterfactual_df = generate_multiple_counterfactuals_for_sample(
                    ppo_model=ppo_model,
                    env=env,
                    dataset_path=st.session_state['confirmed_dataset_path'],
                    sample_index=sample_index,
                    num_counterfactuals=num_counterfactuals,
                    save_path=save_path,
                    max_steps_per_sample=max_steps_per_sample,
                    use_mcts=use_mcts,
                    mcts_simulations=mcts_simulations,
                    verbose=1
                )
            logger.info("Counterfactual generation completed.")

            # Calculate metrics
            logger.info("Calculating metrics...")
            dataset_utils = DatasetUtils(st.session_state['confirmed_dataset_path'], verbose=1)
            metrics = get_metrics(
                original_df=original_df,
                counterfactual_df=counterfactual_df,
                counterfactuals=counterfactuals,
                constraints=st.session_state['temp_constraints'],
                feature_columns=dataset_utils.feature_columns,
                original_data=dataset_utils.dataset,
                verbose=1
            )
            diversity = get_diversity(counterfactual_df) if generation_mode == "Multiple Counterfactuals for a Specific Sample" else None
            logger.info("Metrics calculation completed.")

            # Store results
            st.session_state['counterfactuals'] = counterfactuals
            st.session_state['original_df'] = original_df
            st.session_state['counterfactual_df'] = counterfactual_df
            st.session_state['metrics'] = metrics
            st.session_state['diversity'] = diversity

            st.success("Counterfactual generation completed!")
        except Exception as e:
            st.error(f"Error during counterfactual generation: {e}")
            logger.error(f"Error: {e}")

# Display results if available
if st.session_state['counterfactuals']:
    st.header("Results")
    st.subheader("Original DataFrame")
    st.dataframe(st.session_state['original_df'])
    st.subheader("Counterfactual DataFrame")
    st.dataframe(st.session_state['counterfactual_df'])
    st.subheader("Metrics")
    coverage, distance, implausibility, sparsity, actionability = st.session_state['metrics']
    st.write(f"**Coverage**: {coverage:.2%}")
    st.write(f"**Mean Distance**: {distance:.2f}")
    st.write(f"**Mean Implausibility**: {implausibility:.2f}")
    st.write(f"**Mean Sparsity**: {sparsity:.2f}")
    st.write(f"**Mean Actionability**: {actionability:.2f}")
    if st.session_state['diversity'] is not None:
        st.write(f"**Diversity**: {st.session_state['diversity']:.2f}")

# Sidebar instructions
st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.write("1. Select the generation method from the dropdown menu.")
st.sidebar.write("2. Configure all parameters in the main section.")
st.sidebar.write("3. Load a dataset to enable constraint selection.")
st.sidebar.write("4. Add constraints as needed using the 'Add Constraint' button.")
st.sidebar.write("5. The 'Run' button is enabled only when a valid dataset is loaded and, for single-sample mode, the sample index is valid.")
st.sidebar.write("6. Click 'Run' to generate counterfactuals and view results.")