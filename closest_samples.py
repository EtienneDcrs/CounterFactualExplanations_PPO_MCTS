import numpy as np
import pandas as pd
import pickle

def calculate_distance_simple(a, b, dataset, feature_order):
    total_distance = 0.0
    for i, col in enumerate(feature_order):
        if pd.api.types.is_numeric_dtype(dataset[col]):
            # Numerical: absolute difference
            total_distance += max(float(a[i]) / float(b[i]), float(b[i]) / float(a[i])) if a[i] != 0 and b[i] != 0 else abs(float(a[i]) - float(b[i]))
        else:
            # Categorical: 1 if different, 0 if same
            total_distance += 0 if a[i] == b[i] else 1
    return total_distance

def get_closest_samples(input_sample, dataset, X=5, require_different_outcome=True, show_diff_only=False):
    feature_order = list(dataset.columns[:-1])  # Exclude target
    target_col = dataset.columns[-1]
    # Get the outcome of the input sample

    input_outcome = input_sample[-1] 
    input_features = input_sample[:-1]
    print(f"Input outcome: {input_outcome}, original sample: {input_sample}")

    distances = []
    for idx, row in dataset.iterrows():
        sample_features = row[feature_order].values
        dist = calculate_distance_simple(input_features, sample_features, dataset, feature_order)
        # Only keep samples with a different outcome if required
        if not require_different_outcome or row[target_col] != input_outcome:
            print(f"Sample {idx}: Distance={dist}, Outcome={row[target_col]}, original={input_outcome}")
            distances.append((idx, dist, row[target_col], sample_features))
    distances.sort(key=lambda x: x[1])
    closest = distances[:X]
    closest_indices = [idx for idx, _, _, _ in closest]
    closest_samples = dataset.iloc[closest_indices].copy()
    # Add distance column
    closest_samples['distance'] = [dist for _, dist, _, _ in closest]
    # Prepare original sample as DataFrame
    original_df = pd.DataFrame([list(input_features)], columns=feature_order)
    original_df['distance'] = 0.0
    if target_col not in original_df.columns:
        original_df[target_col] = input_outcome
    # Optionally show only differences
    if show_diff_only:
        diff_rows = []
        for _, _, _, sample_features in closest:
            diff_row = {}
            for i, col in enumerate(feature_order):
                if sample_features[i] == input_features[i]:
                    diff_row[col] = '-'
                else:
                    diff_row[col] = sample_features[i]
            diff_rows.append(diff_row)
        diff_df = pd.DataFrame(diff_rows)
        diff_df['distance'] = [dist for _, dist, _, _ in closest]
        diff_df[target_col] = [outcome for _, _, outcome, _ in closest]
        result_df = pd.concat([original_df, diff_df], ignore_index=True)
    else:
        result_df = pd.concat([original_df, closest_samples], ignore_index=True)
    result_df.to_csv('original_and_closest_samples.csv', index=False)
    print(f"Original and closest {X} samples (with different outcome={require_different_outcome}) saved to 'original_and_closest_samples.csv'")
    return closest_samples

if __name__ == "__main__":
    # Example usage
    # Load dataset and input sample
    dataset_path = 'data/breast_cancer.csv'
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.drop(columns=['capital-gain', 'capital-loss'], errors='ignore')
    input_sample = dataset.iloc[129].values 
    print("Input Sample:", input_sample)
    closest = get_closest_samples(input_sample, dataset, X=25, require_different_outcome=True, show_diff_only=True)
    print(closest)