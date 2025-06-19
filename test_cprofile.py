import csv

def calculate_distance(original_features, modified_features):
    """Calculate the distance between original and modified features using raw features."""
    #         # distance = self.distance(self.original_features, self.modified_features)
    #         dist = 0
    #         for i, feature in enumerate(self.modified_features):
    #             dist += abs(self.original_features[i] - feature)
    #         return dist
    """
    Calculate hybrid distance that treats categorical and continuous features differently.
    
    - Categorical features: 100% change if different, 0% if same
    - Continuous features: Relative percentage change normalized by feature range
    
    Args:
        original_features: Original feature values (raw, not encoded)
        modified_features: Modified feature values (raw, not encoded)
    
    Returns:
        float: Weighted distance score
    """
    if len(original_features) != len(modified_features):
        raise ValueError("Feature vectors must have the same length")
    
    total_distance = 0.0
    num_features = len(original_features)
    
    for i in range(num_features):
        orig_val = original_features[i]
        mod_val = modified_features[i]
        
        # Determine if the feature is categorical by checking its type
        if isinstance(orig_val, str) or isinstance(orig_val, bool):
            # Categorical feature: 100% change if different, 0% if same
            if orig_val != mod_val:
                total_distance += 1.0
        else:
            # Continuous feature: relative percentage change
            relative_change = max(mod_val, orig_val) / min(mod_val, orig_val) if min(mod_val, orig_val) != 0 else 1.0
            total_distance += relative_change
    
    return total_distance

if __name__ == "__main__":
    # Paths to the CSV files
    original_csv = "data/generated_counterfactuals_ppo_drug200_original.csv"
    counterfactual_csv = "data/generated_counterfactuals_ppo_drug200_counterfactual.csv"

    def read_first_row(csv_path):
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header
            row = next(reader)
            # Try to convert to float if possible, else keep as string
            def parse_value(val):
                try:
                    return float(val)
                except ValueError:
                    if val.lower() == 'true':
                        return True
                    elif val.lower() == 'false':
                        return False
                    return val
            return [parse_value(v) for v in row]

    original_features = read_first_row(original_csv)
    modified_features = read_first_row(counterfactual_csv)

    distance = calculate_distance(original_features, modified_features)
    print(f"Distance between original and counterfactual: {distance}")