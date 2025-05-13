import numpy as np
import torch
import gymnasium as gym
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import joblib
import os
from Classifier_model import Classifier 

# Import your MCTS implementation
from PPO_MCTS_CFE import CounterfactualMCTS, MCTSPolicyWrapper, integrate_mcts_with_ppo

def main():
    """
    Main function to demonstrate PPO-MCTS for counterfactual explanations
    """
    print("Starting PPO-MCTS Counterfactual Explanation Demo")
    
    # 1. Load your trained PPO model (.zip format is standard for SB3)
    model_path = "ppo_certifai_final_diabetes.zip"  # Update this path to your model location
    try:
        ppo_model = PPO.load(model_path)
        print(f"Successfully loaded PPO model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure your model is a valid Stable Baselines3 PPO model")
        return
    
    # 2. Load or create your classifier
    # This is the model we want to explain (generate counterfactuals for)
    try:
        classifier_path = "classification_models/diabetes_model.pt"  
        classifier = Classifier(in_feats=8, out=2)
        classifier.load_state_dict(torch.load(classifier_path))
        classifier.eval()  # Set to evaluation mode
        print(f"Successfully loaded classifier from {classifier_path}")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        print("No classifier found, creating a dummy classifier for demonstration")
        # Creating a dummy classifier class for demonstration
        class DummyClassifier:
            def predict(self, X):
                # Simple example: classify points based on their first dimension
                return [1 if x[0] > 0 else 0 for x in X]
        classifier = DummyClassifier()
    
    # 3. Load dataset and calculate feature ranges (min and max values for each feature)
    dataset_path = 'data/diabetes.csv'
    try:
        print(f"Loading dataset from {dataset_path} to determine feature ranges...")
        df = pd.read_csv(dataset_path)
        
        # Get feature columns (exclude target column, assumed to be the last one)
        feature_columns = df.columns[:-1]
        print(f"Found {len(feature_columns)} features: {', '.join(feature_columns)}")
        
        # Calculate min and max for each feature
        feature_ranges = []
        for col in feature_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            # Add a small buffer to ranges (5%)
            buffer = (max_val - min_val) * 0.05
            feature_ranges.append((min_val - buffer, max_val + buffer))
            print(f"Feature '{col}' range: ({min_val:.2f}, {max_val:.2f})")
        
        # Also get a sample from the dataset to use as original instance
        sample_idx = 0  # Can be random or a specific index
        original_instance = df.iloc[sample_idx, :-1].values
        print(f"Using sample at index {sample_idx} as original instance")
        
    except Exception as e:
        print(f"Error loading dataset or calculating feature ranges: {e}")
        print("Using default feature ranges instead")
        feature_ranges = [
            (-5, 5),  # Feature 0 range
            (-5, 5),  # Feature 1 range
            (-5, 5),  # Feature 2 range
            (-5, 5),  # Feature 3 range
        ]
        # Default instance
        original_instance = np.array([6, 148, 72, 35, 0, 33.6, 0.627, 50])
    
    # Get the actual feature dimension from the policy
    try:
        feature_dim_ppo = ppo_model.policy.observation_space.shape[0]
        print(f"Detected feature dimension from policy: {feature_dim_ppo}")
        
        # Create extended feature ranges to match actual dimensions for PPO
        # We will use original 8 feature ranges for the classifier separately
        extended_feature_ranges = feature_ranges.copy()
        if len(extended_feature_ranges) < feature_dim_ppo:
            print(f"Extending feature ranges to match PPO feature dimension {feature_dim_ppo}")
            # For original features, we'll use the actual ranges
            # For modified features (next 8 features), use the same ranges
            # For metadata (last 3 features), use (0,1) range
            extended_feature_ranges.extend(feature_ranges)  # Add ranges again for modified features
            extended_feature_ranges.extend([(0, 1)] * 3)  # Add ranges for metadata
    except:
        print("Could not determine feature dimension from policy. Using calculated ranges.")
        extended_feature_ranges = feature_ranges.copy()
    
    # Create a wrapper class to handle the different dimensions
    class ClassifierWrapper:
        def __init__(self, classifier, original_instance):
            self.classifier = classifier
            self.original_instance = original_instance
            
        def predict(self, X):
            # Extract only the first 8 features (original features) from the PPO state representation
            if len(X[0]) > 8:
                X_classifier = [x[:8] for x in X]  # Get only the original features
            else:
                X_classifier = X
                
            # Convert to tensor for classifier
            X_tensor = torch.tensor(X_classifier, dtype=torch.float)
            
            # Get prediction
            with torch.no_grad():
                output = self.classifier(X_tensor)
                predictions = output.argmax(dim=-1).cpu().numpy()
                
            return predictions
    
    # Wrap the classifier to handle dimension differences
    classifier_wrapper = ClassifierWrapper(classifier, original_instance)
    
    # 5. Get current prediction for the original instance
    model_input = torch.tensor(original_instance, dtype=torch.float32)
    with torch.no_grad():
        original_output = classifier(model_input)
        original_prediction = original_output.argmax().item()
    print(f"Original instance: {original_instance}")
    print(f"Original prediction: {original_prediction}")
    
    # 6. Define target class for counterfactual
    # The goal is to find an instance similar to original_instance but classified as target_class
    target_class = 1 - original_prediction  # Toggle class (0->1, 1->0)
    print(f"Target class for counterfactual: {target_class}")
    
    # 7. Create MCTS wrapper or integrate MCTS with PPO
    print("Integrating MCTS with PPO model...")
    
    # Create a custom action mapper to handle the dimension differences
    def action_mapper(state, action):
        """Maps abstract actions to feature modifications on the original 8 features"""
        new_state = state.copy()
        
        # Extract feature index and direction from action
        feature_idx = action // 2
        direction = 1 if action % 2 == 0 else -1
        
        # Only modify if within the original 8 features
        if feature_idx < 8:
            # Get step size based on feature range
            step_size = (extended_feature_ranges[feature_idx][1] - extended_feature_ranges[feature_idx][0]) / 20.0
            
            # Modify the feature
            new_state[feature_idx] += direction * step_size
            
            # Apply bounds
            new_state[feature_idx] = max(extended_feature_ranges[feature_idx][0], 
                                        min(extended_feature_ranges[feature_idx][1], 
                                            new_state[feature_idx]))
            
            # If we have modified features in state representation, update them too
            if len(new_state) > 8:
                new_state[8 + feature_idx] = new_state[feature_idx]
                
        return new_state
    
    # Create extended state for PPO
    def create_extended_state(original_instance):
        """Create a state representation compatible with the PPO model"""
        if len(original_instance) == feature_dim_ppo:
            return original_instance  # Already in correct format
        
        # Create extended state: original + modified + metadata
        extended_state = np.zeros(feature_dim_ppo)
        
        # Copy original features
        n_original = min(len(original_instance), 8)
        extended_state[:n_original] = original_instance[:n_original]
        
        # Copy same values to modified features section
        extended_state[8:8+n_original] = original_instance[:n_original]
        
        # Initialize metadata (steps, distance, prediction)
        extended_state[-3:] = [0, 0, original_prediction]
        
        return extended_state
    
    # Convert original instance to extended state for PPO
    extended_state = create_extended_state(original_instance)
    
    mcts_enhanced_model, original_policy = integrate_mcts_with_ppo(
        env=None,  # Not needed for inference
        ppo_model=ppo_model,
        classifier=classifier_wrapper,
        feature_ranges=extended_feature_ranges,
        target_class=target_class
    )
    
    # 8. Set up the original instance for counterfactual search
    mcts_enhanced_model.policy.mcts.original_instance = original_instance.copy()  # Keep original 8 features as reference
    mcts_enhanced_model.policy.mcts.action_mapper = action_mapper  # Use custom action mapper
    
    # 9. Find the best counterfactual
    print("Searching for counterfactual explanation...")
    counterfactual, reward = mcts_enhanced_model.policy.mcts.get_best_counterfactual(max_steps=50)
    
    # 10. Verify and output results
    if counterfactual is not None:
        # Extract first 8 features if counterfactual has more
        counterfactual_for_classifier = counterfactual[:8] if len(counterfactual) > 8 else counterfactual
        
        # Get prediction for counterfactual
        cf_prediction = classifier_wrapper.predict([counterfactual_for_classifier])[0]
        
        print("\nCounterfactual found!")
        print(f"Counterfactual instance: {counterfactual_for_classifier}")
        print(f"Counterfactual prediction: {cf_prediction}")
        print(f"Target class: {target_class}")
        print(f"Reward: {reward}")
        
        # Calculate changes
        changes = np.count_nonzero(counterfactual_for_classifier != original_instance[:8])
        distance = np.linalg.norm(counterfactual_for_classifier - original_instance[:8])
        print(f"L2 distance from original: {distance:.4f}")
        print(f"Number of features changed: {changes}")
        
        # Show specific changes
        print("\nFeature changes:")
        for i in range(len(original_instance[:8])):
            if original_instance[i] != counterfactual_for_classifier[i]:
                feature_name = feature_columns[i] if i < len(feature_columns) else f"Feature {i}"
                print(f"{feature_name}: {original_instance[i]:.4f} -> {counterfactual_for_classifier[i]:.4f} " +
                      f"(delta: {counterfactual_for_classifier[i] - original_instance[i]:.4f})")
    else:
        print("No valid counterfactual found within the specified steps.")
    
    # 11. Restore original policy if needed
    ppo_model.policy = original_policy
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()