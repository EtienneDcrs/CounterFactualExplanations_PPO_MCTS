from collections import OrderedDict
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import os

class Classifier(pl.LightningModule):
    def __init__(self, in_feats=8, h_size=64, out=2, n_layers=3,
                 activation_function=nn.ReLU, lr=1e-3, dropout=0.3, weight_decay=1e-4):
        super().__init__()

        mlp_sequence = OrderedDict([('mlp1', nn.Linear(in_features=in_feats,
                                                       out_features=h_size)),
                                    ('activ1', activation_function()),
                                    ('dropout1', nn.Dropout(dropout))])

        for i in range(1, n_layers):
            new_keys = ['mlp' + str(i + 1), 'activ' + str(i + 1), 'dropout' + str(i + 1)]
            mlp_sequence[new_keys[0]] = nn.Linear(in_features=h_size,
                                                  out_features=h_size)
            mlp_sequence[new_keys[1]] = activation_function()
            mlp_sequence[new_keys[2]] = nn.Dropout(dropout)

        mlp_sequence['out_projection'] = nn.Linear(in_features=h_size,
                                                   out_features=out)

        self.net = nn.Sequential(mlp_sequence)
        self.learning_rate = lr
        self.weight_decay = weight_decay

    def forward(self, x, apply_softmax=False):
        y = self.net(x)
        if apply_softmax:
            return nn.functional.softmax(y, -1)
        return y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        
        # Calculate accuracy
        preds = torch.argmax(yhat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        
        # Calculate accuracy
        preds = torch.argmax(yhat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)


def preprocess_categorical_data(dataset):
    """
    Convert categorical data to numerical format using label encoding
    """
    from sklearn.preprocessing import StandardScaler
    
    dataset_encoded = dataset.copy()
    label_encoders = {}
    
    for column in dataset_encoded.columns:
        if dataset_encoded[column].dtype == 'object':
            le = LabelEncoder()
            dataset_encoded[column] = le.fit_transform(dataset_encoded[column])
            label_encoders[column] = le
    
    # Normalize numerical features (except target column)
    scaler = StandardScaler()
    feature_columns = dataset_encoded.columns[:-1]  # All except last (target) column
    dataset_encoded[feature_columns] = scaler.fit_transform(dataset_encoded[feature_columns])
    
    return dataset_encoded, label_encoders, scaler


def train_model(dataset_path, model_path):
    # Load dataset - always assume first row is header
    dataset = pd.read_csv(dataset_path, header=0)
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"First few rows:\n{dataset.head()}")
    print(f"Unique values in target column (last column): {dataset.iloc[:, -1].unique()}")
    
    # Preprocess categorical data
    dataset_encoded, label_encoders, scaler = preprocess_categorical_data(dataset)
    
    # Convert DataFrame to numpy array first, then to tensor
    try:
        model_input = torch.tensor(dataset_encoded.values, dtype=torch.float)
    except Exception as e:
        print(f"Error converting to tensor: {e}")
        print(f"Dataset dtypes: {dataset_encoded.dtypes}")
        print(f"Dataset shape: {dataset_encoded.shape}")
        print(f"First few rows:\n{dataset_encoded.head()}")
        raise
    
    # Extract predictors (all columns except the last one, since target is last column)
    predictors = model_input[:, :-1]
    
    # Extract target (the last column) - ensure it's long/int type for classification
    target = model_input[:, -1].long()
    
    # Print target information for debugging
    print(f"Target shape: {target.shape}")
    print(f"Unique target values: {torch.unique(target)}")
    print(f"Number of classes: {len(torch.unique(target))}")
    print(f"Class distribution: {torch.bincount(target)}")
    
    # Ensure target classes are 0-indexed
    unique_targets = torch.unique(target)
    if not torch.equal(unique_targets, torch.arange(len(unique_targets))):
        print("Remapping target values to be 0-indexed...")
        target_mapping = {old_val.item(): new_val for new_val, old_val in enumerate(unique_targets)}
        for old_val, new_val in target_mapping.items():
            target[target == old_val] = new_val
        print(f"Target mapping: {target_mapping}")
        print(f"New unique target values: {torch.unique(target)}")

    # Improved training parameters
    val_percentage = 0.2  # Increased validation split
    batch_size = 32  # Increased batch size
    max_epochs = 500  # More epochs
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,  # More patience
        strict=False,
        verbose=True,
        mode='min'
    )

    # Shuffle the data before splitting
    indices = torch.randperm(len(predictors))
    predictors = predictors[indices]
    target = target[indices]
    
    split_idx = int(len(predictors) * (1 - val_percentage))
    
    train_loader = DataLoader(
        TensorDataset(predictors[:split_idx], target[:split_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        TensorDataset(predictors[split_idx:], target[split_idx:]),
        batch_size=batch_size,
        num_workers=0
    )
    
    # Create model with improved architecture
    in_features = predictors.shape[1]
    num_classes = len(torch.unique(target))
    print(f"Creating model with {in_features} input features and {num_classes} output classes")
    
    # Improved model parameters
    model = Classifier(
        in_feats=in_features, 
        out=num_classes,
        h_size=128,      # Larger hidden size
        n_layers=4,      # More layers
        dropout=0.3,     # Dropout for regularization
        lr=0.001,        # Learning rate
        weight_decay=1e-4 # L2 regularization
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        callbacks=[early_stop],
        accelerator='auto',
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Trained classifier model saved to {model_path}")
    
    # Save label encoders and scaler
    import pickle
    encoders_path = model_path.replace('.pt', '_encoders.pkl')
    scaler_path = model_path.replace('.pt', '_scaler.pkl')
    
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Label encoders saved to {encoders_path}")
    print(f"Scaler saved to {scaler_path}")
    
    return model, label_encoders, scaler


def load_classifier_with_preprocessing(dataset_path, model_path=None):
    """
    Updated version of your original loading code that handles categorical data
    """
    # Load the dataset - always assume first row is header
    dataset = pd.read_csv(dataset_path, header=0)
    
    # Preprocess categorical data to get the correct dimensions
    dataset_encoded, _, _ = preprocess_categorical_data(dataset)
    
    # Determine the output size based on the last column (target)
    out = len(dataset_encoded.iloc[:, -1].unique())
    print(f"Detected {out} output classes")
    
    # Load or train the classifier model
    if model_path is None:
        model_path = f"{os.path.splitext(os.path.basename(dataset_path))[0]}_model.pt"
        model_path = os.path.join("classification_models", model_path)
    
    # Check if the classifier model exists, otherwise train a new one
    if not os.path.exists(model_path):
        print(f"Classifier model not found at {model_path}. Training a new one.")
        train_model(dataset_path, model_path)
        # Reload the dataset after training to ensure consistency
        dataset_encoded, _, _ = preprocess_categorical_data(dataset)
    
    print(f"Loading classifier model from {model_path}")
    
    # Determine input features (all columns except the last one which is target)
    in_feats = dataset_encoded.shape[1] - 1
    print(f"Model architecture: {in_feats} input features, {out} output classes")
    
    # Load the classifier model with improved architecture
    classifier = Classifier(
        in_feats=in_feats, 
        out=out,
        h_size=128,      # Match training parameters
        n_layers=4,
        dropout=0.3,
        lr=0.001,
        weight_decay=1e-4
    )
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()  # Set to evaluation mode
    
    # Load label encoders and scaler if available
    encoders_path = model_path.replace('.pt', '_encoders.pkl')
    scaler_path = model_path.replace('.pt', '_scaler.pkl')
    
    label_encoders = None
    scaler = None
    
    try:
        import pickle
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        print(f"Label encoders loaded from {encoders_path}")
    except FileNotFoundError:
        print(f"Warning: Label encoders not found at {encoders_path}")
        print("You may need to retrain the model to save the encoders.")
    
    try:
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded from {scaler_path}")
    except FileNotFoundError:
        print(f"Warning: Scaler not found at {scaler_path}")
        print("You may need to retrain the model to save the scaler.")
    
    return classifier, label_encoders, scaler


def test_model_accuracy(model, dataset_path, label_encoders=None, scaler=None, test_split=0.2):
    """
    Test the accuracy of a trained model on a dataset
    
    Args:
        model: Trained classifier model
        dataset_path: Path to the dataset CSV file
        label_encoders: Dictionary of label encoders (optional, will create new ones if None)
        scaler: StandardScaler object (optional, will create new one if None)
        test_split: Fraction of data to use for testing (default: 0.2)
    
    Returns:
        accuracy: Test accuracy as a float
        predictions: Model predictions
        true_labels: True labels
    """
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    
    # Load dataset - always assume first row is header
    dataset = pd.read_csv(dataset_path, header=0)
    
    # Preprocess categorical data
    if label_encoders is None or scaler is None:
        dataset_encoded, _, _ = preprocess_categorical_data(dataset)
    else:
        # Use existing label encoders and scaler
        dataset_encoded = dataset.copy()
        for column in dataset_encoded.columns:
            if dataset_encoded[column].dtype == 'object' and column in label_encoders:
                try:
                    dataset_encoded[column] = label_encoders[column].transform(dataset_encoded[column])
                except ValueError as e:
                    print(f"Warning: Unknown category in column {column}. Using fit_transform instead.")
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    dataset_encoded[column] = le.fit_transform(dataset_encoded[column])
        
        # Apply scaling to features (except target column)
        if scaler is not None:
            feature_columns = dataset_encoded.columns[:-1]
            dataset_encoded[feature_columns] = scaler.transform(dataset_encoded[feature_columns])
    
    # Convert to tensor
    model_input = torch.tensor(dataset_encoded.values, dtype=torch.float)
    
    # Extract predictors and target
    predictors = model_input[:, :-1]
    target = model_input[:, -1].long()
    
    # Split data for testing
    if test_split > 0:
        # Use the same random seed for reproducible splits
        torch.manual_seed(42)
        indices = torch.randperm(len(predictors))
        split_idx = int(len(predictors) * (1 - test_split))
        test_indices = indices[split_idx:]
        test_predictors = predictors[test_indices]
        test_target = target[test_indices]
    else:
        # Use entire dataset for testing
        test_predictors = predictors
        test_target = target
    
    print(f"Testing on {len(test_predictors)} samples...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        logits = model(test_predictors)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_target.cpu().numpy(), predictions.cpu().numpy())
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_target.cpu().numpy(), predictions.cpu().numpy()))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_target.cpu().numpy(), predictions.cpu().numpy())
    print(cm)
    
    return accuracy, predictions.cpu().numpy(), test_target.cpu().numpy()


def evaluate_model_on_full_dataset(model, dataset_path, label_encoders=None, scaler=None):
    """
    Evaluate model accuracy on the entire dataset (useful for checking training performance)
    
    Args:
        model: Trained classifier model
        dataset_path: Path to the dataset CSV file
        label_encoders: Dictionary of label encoders
        scaler: StandardScaler object
    
    Returns:
        accuracy: Full dataset accuracy as a float
    """
    print("Evaluating model on full dataset...")
    accuracy, predictions, true_labels = test_model_accuracy(
        model, dataset_path, label_encoders, scaler, test_split=0.0
    )
    return accuracy