
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import manhattan_distances as L1
from sklearn.metrics.pairwise import euclidean_distances as L2


def normalize_value(value, min_value, max_value):
    """
    Normalize the value to a value between 0 and 1.
    """
    if max_value - min_value == 0:
        return 0
    return (float(value) - float(min_value)) / (float(max_value) - float(min_value))



def proximity_KPI(x, y, con=None, cat=None):
    """
    Calculate the average distance between the true and predicted values.
    All the values are first normalized to a value between 0 and 1.

    Parameters:
    - x: The original DataFrame.
    - y: The DataFrame of counterfactual explanations.
    - con: List of continuous features.
    - cat: List of categorical features.

    Returns:
    - The average distance between the true and predicted values.
    """
    assert isinstance(x, pd.DataFrame), 'This distance can be used only if input is a row of a pandas DataFrame at the moment.'

    # Convert y to DataFrame if it's not already
    if not isinstance(y, pd.DataFrame):
        # Check if y is a numpy array or list
        if isinstance(y, (np.ndarray, list)):
            # Create DataFrame with the same columns as x
            y = pd.DataFrame(y, columns=x.columns.tolist())
        else:
            raise ValueError("Counterfactual samples must be a DataFrame, numpy array, or list")


    # Ensure con and cat are lists
    con = con or []
    cat = cat or []

    # Calculate min and max for normalization
    mins = {col: min(float(x[col].min()), float(y[col].min())) for col in x.columns if col in con}
    maxs = {col: max(float(x[col].max()), float(y[col].max())) for col in x.columns if col in con}
    distances = []  # list to store the distances

    # Calculate distances
    for i in range(len(x)):
        distance = 0  # initialize the sample's distance to 0

        for col in x.columns:
            if col in cat:
                # calculate the distance for categorical features
                if x[col].iloc[i] != y[col].iloc[i]:
                    distance += 1
            else:
                # calculate the distance for continuous features
                normalized_x = normalize_value(x[col].iloc[i], mins[col], maxs[col])
                normalized_y = normalize_value(y[col].iloc[i], mins[col], maxs[col])
                distance += (normalized_y - normalized_x) ** 2
            # else:
            #     raise ValueError(f"Column {col} is not specified as continuous or categorical.")

        distance = distance ** 0.5
        distances.append(distance)  # append the distance to the list

    average_distance = sum(distances) / len(distances)  # calculate the average distance
    return round(average_distance, 4)  

def sparsity_KPI(x, y):
    """
    Calculate the average sparsity between the true and predicted values.
    Sparsity is defined as the number of features that are changed.

    Parameters:
    - x: The original DataFrame.
    - y: The DataFrame of counterfactual explanations.

    Returns:
    - The average sparsity of the counterfactuals.
    """
    assert isinstance(x, pd.DataFrame), 'This sparsity metric can be used only if input is a row of a pandas DataFrame at the moment.'

    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y, columns=x.columns.tolist())
    else:
        y.columns = x.columns.tolist()

    sparsities = []  # list to store the sparsity values

    # Calculate sparsity for each instance
    for i in range(len(x)):
        changes = 0  # initialize the number of changes to 0

        for col in x.columns:
            if x[col].iloc[i] != y[col].iloc[i]:
                changes += 1

        sparsities.append(changes)  # append the number of changes to the list

    average_sparsity = sum(sparsities) / len(sparsities)  # calculate the average sparsity
    return round(average_sparsity, 4)


def validity_KPI(model, x, y, desired_outcome=None, device='cpu'):
    """
    Calculate the validity of counterfactual explanations.
    Validity is defined as the percentage of counterfactuals that flip the model's prediction to the desired outcome.

    Parameters:
    - model: The trained PyTorch model used for predictions.
    - x: The original DataFrame.
    - y: The DataFrame of counterfactual explanations.
    - desired_outcome: The desired outcome for the counterfactuals.
    - device: The device to run the model on ('cpu' or 'cuda').

    Returns:
    - The percentage of valid counterfactuals.
    """
    assert isinstance(x, pd.DataFrame), 'This validity metric can be used only if input is a row of a pandas DataFrame at the moment.'

    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y, columns=x.columns.tolist())
    else:
        y.columns = x.columns.tolist()

    # Ensure the model is in evaluation mode
    model.eval()

    # Move the model to the specified device
    model.to(device)

    # Convert DataFrames to tensors
    x_tensor = torch.tensor(x.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

    # Get predictions for the original and counterfactual instances
    with torch.no_grad():
        original_predictions = model(x_tensor).argmax(dim=1).cpu().numpy()
        counterfactual_predictions = model(y_tensor).argmax(dim=1).cpu().numpy()

    # Count the number of valid counterfactuals
    if desired_outcome is not None:
        valid_counterfactuals = sum(counterfactual_predictions == desired_outcome)
    else:
        # If no desired outcome is provided, count all counterfactuals as valid if they differ from the original instance
        valid_counterfactuals = sum(counterfactual_predictions != original_predictions)

    # Calculate the percentage of valid counterfactuals
    validity_percentage = (valid_counterfactuals / len(y)) * 100

    return round(validity_percentage, 2)



if __name__ == "__main__":
    # Example usage
    x = pd.DataFrame({'age': [25, 30, 45], 'income': [30000, 50000, 70000], 'sex': ['F', 'F', 'M']})
    y = pd.DataFrame({'age': [39, 32, 48], 'income': [32000, 62000, 72000], 'sex': ['M', 'F', 'F']})
    con = ['age', 'income']
    cat = ['sex']
    print(proximity_KPI(x, y, con, cat))
    print(sparsity_KPI(x, y))

