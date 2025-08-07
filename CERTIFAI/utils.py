import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify categorical columns in a DataFrame based on dtype.
    
    Args:
        df: DataFrame to analyze.
    
    Returns:
        List of column names that are categorical (object or category dtype).
    """
    return [col for col in df.columns if df[col].dtype in ['object', 'category']]

def calculate_distance(original_features: np.ndarray, modified_features: np.ndarray, 
                      categorical_columns: List[str], feature_columns: List[str]) -> float:
    """
    Calculate L1 (Manhattan) distance between encoded original and modified features.
    
    Args:
        original_features: Original feature vector.
        modified_features: Modified feature vector.
        categorical_columns: List of categorical column names.
        feature_columns: List of all feature column names.
    
    Returns:
        L1 distance between the feature vectors.
    """
    categorical_indices = [feature_columns.index(col) for col in categorical_columns if col in feature_columns]
    dist = 0
    for i, (o, m) in enumerate(zip(original_features, modified_features)):
        if i in categorical_indices:
            dist += float(o != m)
        else:
            try:
                dist += abs(float(o) - float(m))
            except (ValueError, TypeError):
                logger.warning(f"Non-numeric value encountered in numerical feature at index {i} ({feature_columns[i]}). Treating as categorical.")
                dist += float(o != m)
    return dist

def get_closest_samples(input_features: np.ndarray, dataset: pd.DataFrame, X: int = 5, 
                        feature_columns: List[str] = None, categorical_columns: List[str] = None) -> pd.DataFrame:
    """
    Find the X closest samples to the input features in the original dataset.
    
    Args:
        input_features: Input feature vector (numpy array).
        dataset: DataFrame containing the original dataset.
        X: Number of closest samples to return (default: 5).
        feature_columns: List of feature column names to use for distance calculation.
        categorical_columns: List of categorical column names.
    
    Returns:
        DataFrame containing the X closest samples.
    """
    if feature_columns is None:
        feature_columns = dataset.columns.tolist()
    
    if categorical_columns is None:
        categorical_columns = get_categorical_columns(dataset)
    
    # Ensure input_features is a 2D array for distance calculation
    input_features = np.array(input_features).reshape(1, -1)
    
    # Extract features from dataset
    data_features = dataset[feature_columns].values
    
    # Compute L1 (Manhattan) distances
    distances = [calculate_distance(input_features[0], sample, categorical_columns, feature_columns) 
                 for sample in data_features]
    
    # Get indices of sorted distances
    sorted_indices = np.argsort(distances)[:X]
    
    # Return the closest samples
    closest_samples = dataset.iloc[sorted_indices].copy()
    closest_samples['distance'] = [distances[idx] for idx in sorted_indices]
    return closest_samples

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

def get_diversity(counterfactual_df: pd.DataFrame, feature_columns: List[str], 
                  categorical_columns: List[str]) -> float:
    """
    Calculate diversity of counterfactuals for a unique sample. 
    Diversity is the smallest distance of a counterfactual with its furthest neighbor.
    
    Args:
        counterfactual_df: DataFrame of counterfactual samples.
        feature_columns: List of feature column names.
        categorical_columns: List of categorical column names.
    
    Returns:
        Diversity score (higher value means better diversity).
    """
    if counterfactual_df.empty:
        return 0.0
    
    diversities = []
    num_counterfactuals = len(counterfactual_df)

    # For each counterfactual, calculate the distance to all others and store the maximum distance
    for i in range(num_counterfactuals):
        distances = []
        for j in range(num_counterfactuals):
            if i != j:
                dist = calculate_distance(counterfactual_df[feature_columns].iloc[i].values, 
                                        counterfactual_df[feature_columns].iloc[j].values,
                                        categorical_columns, feature_columns)
                distances.append(dist)
        if distances:
            diversities.append(min(distances))
    
    return round(min(diversities), 2) if diversities else 0.0

def get_metrics(original_csv_path: str, counterfactual_csv_path: str, constraints: Dict[str, str]) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate KPIs for counterfactuals based on CSV files of original and counterfactual features.
    
    Args:
        original_csv_path: Path to CSV file containing original feature samples (with 'sample_id' column).
        counterfactual_csv_path: Path to CSV file containing counterfactual feature samples (with 'sample_id' and 'counterfactual_found' columns).
        constraints: Dictionary of feature constraints (e.g., {"sex": "fixed"}).
    
    Returns:
        Tuple of (coverage, distance, implausibility, sparsity, actionability, diversity).
    """
    logger.debug(f"Calculating KPIs for counterfactuals from {counterfactual_csv_path}...")

    # Load CSV files
    try:
        original_df = pd.read_csv(original_csv_path)
        counterfactual_df = pd.read_csv(counterfactual_csv_path)
    except Exception as e:
        logger.error(f"Failed to load CSV files: {e}")
        return 0, 0, 0, 0, 0, 0

    # Validate presence of required columns
    if 'sample_id' not in original_df.columns:
        logger.error("Original CSV must contain 'sample_id' column")
        return 0, 0, 0, 0, 0, 0
    
    if 'sample_id' not in counterfactual_df.columns or 'counterfactual_found' not in counterfactual_df.columns:
        logger.error("Counterfactual CSV must contain 'sample_id' and 'counterfactual_found' columns")
        return 0, 0, 0, 0, 0, 0

    # Get feature columns (excluding metadata)
    feature_columns = [col for col in original_df.columns if col != 'sample_id']
    counterfactual_feature_columns = [col for col in counterfactual_df.columns if col not in ['sample_id', 'counterfactual_found']]

    # Validate feature columns match
    if feature_columns != counterfactual_feature_columns:
        logger.error("Original and counterfactual CSVs must have the same feature columns")
        return 0, 0, 0, 0, 0, 0

    # Validate sample_id alignment
    if not original_df['sample_id'].equals(counterfactual_df['sample_id']):
        logger.error("Original and counterfactual CSVs must have matching 'sample_id' values")
        return 0, 0, 0, 0, 0, 0

    # Get categorical columns
    categorical_columns = get_categorical_columns(original_df[feature_columns])

    # Initialize KPI lists
    distances = []
    implausibility_scores = []
    sparsities = []
    actionability_scores = []
    success_count = 0

    # Process each sample
    for i in range(len(original_df)):
        original_features = original_df[feature_columns].iloc[i].values
        counterfactual_features = counterfactual_df[feature_columns].iloc[i].values
        counterfactual_found = counterfactual_df['counterfactual_found'].iloc[i]
        
        # Skip if no valid counterfactual
        if counterfactual_found != 1:
            continue
        
        success_count += 1
        
        # Calculate distance
        dist = calculate_distance(original_features, counterfactual_features, categorical_columns, feature_columns)
        distances.append(dist)
        
        # Calculate implausibility
        closest_sample = get_closest_samples(counterfactual_features, original_df[feature_columns], 
                                            X=5, feature_columns=feature_columns, 
                                            categorical_columns=categorical_columns).iloc[0]
        implausibility_scores.append(calculate_distance(counterfactual_features, 
                                                      closest_sample[feature_columns].values,
                                                      categorical_columns, feature_columns))
        
        # Calculate sparsity
        changes = sum(1 for j, col in enumerate(feature_columns) 
                     if original_features[j] != counterfactual_features[j])
        sparsities.append(changes)
        
        # Calculate actionability
        actionability_scores.append(calculate_actionability(original_features, counterfactual_features, 
                                                          constraints, feature_columns))
    
    # Calculate KPIs
    num_samples = len(original_df)
    coverage = success_count / num_samples if num_samples > 0 else 0
    distance = np.mean(distances) if distances else 0
    implausibility = np.mean(implausibility_scores) if implausibility_scores else 0
    sparsity = np.mean(sparsities) if sparsities else 0
    actionability = np.mean(actionability_scores) if actionability_scores else 0
    
    # Calculate diversity using only valid counterfactuals
    valid_counterfactual_df = counterfactual_df[counterfactual_df['counterfactual_found'] == 1][feature_columns]
    diversity = get_diversity(valid_counterfactual_df, feature_columns, categorical_columns)
    
    logger.info(f"Coverage: {coverage:.2%} ({success_count}/{num_samples})")
    logger.info(f"Mean distance: {distance:.2f}")
    logger.info(f"Mean implausibility: {implausibility:.2f}")
    logger.info(f"Mean sparsity: {sparsity:.2f}")
    logger.info(f"Mean actionability: {actionability:.2f}")
    logger.info(f"Diversity: {diversity:.2f}")
    
    return coverage, distance, implausibility, sparsity, actionability, diversity
if __name__ == "__main__":
    # Example usage
    dataset = "bank"
    
    original_csv = f"data/generated_counterfactuals_ppo_{dataset}_original.csv"
    counterfactual_csv = f"data/generated_counterfactuals_ppo_{dataset}.csv"
    constraints = {}
    coverage, distance, implausibility, sparsity, actionability, diversity = get_metrics(original_csv, counterfactual_csv, constraints)
    print(f"Coverage: {coverage:.2%}, Distance: {distance:.2f}, Implausibility: {implausibility:.2f}, "
          f"Sparsity: {sparsity:.2f}, Actionability: {actionability:.2f}, Diversity: {diversity:.2f}")