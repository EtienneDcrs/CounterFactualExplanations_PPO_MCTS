
import dice_ml
from dice_ml.utils import helpers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
import pickle

# Fonction pour calculer la distance de Manhattan (L1)
def calculate_distance(original_features, modified_features):
    """
    Calculate L1 (Manhattan) distance between encoded original and modified features.
    """
    categorical_indices = []
    for i, feature in enumerate(original_features):
        if isinstance(feature, str) or isinstance(feature, bool):
            categorical_indices.append(i)
    dist = 0
    for i, (o, m) in enumerate(zip(original_features, modified_features)):
        if i in categorical_indices:
            dist += float(o != m)  # 1 if changed
        else:
            dist += abs(o - m)
    return dist

# Fonction pour calculer la sparsity (nombre de caractéristiques modifiées)
def sparsity(instance, counterfactual):
    modified_features = 0
    for i, (o, m) in enumerate(zip(instance, counterfactual)):
        if o != m:
            modified_features += 1
    return modified_features

# Charger le dataset à partir d'un fichier CSV
def load_custom_dataset(file_path, target_column):
    """
    Load dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        target_column (str): Name of the target/outcome column.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    dataset = pd.read_csv(file_path)
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    return dataset

# Charger les encodeurs et scalers
def load_preprocessors(encoder_path, scaler_path):
    """
    Load encoders and scalers from .pkl files.
    
    Args:
        encoder_path (str): Path to the encoder .pkl file.
        scaler_path (str): Path to the scaler .pkl file.
    
    Returns:
        tuple: Encoder and scaler objects.
    """
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return encoder, scaler

# Définir l'architecture du modèle (copiée de Classifier_model.py)
class Classifier(pl.LightningModule):
    def __init__(self, in_feats=8, h_size=128, out=2, n_layers=4,
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

# Classe pour envelopper le modèle PyTorch pour DiCE
class PyTorchModelWrapper:
    def __init__(self, model_path, encoder, scaler, feature_names, continuous_features):
        # Initialiser le modèle avec les paramètres utilisés lors de l'entraînement
        input_dim = len(feature_names)
        self.model = Classifier(in_feats=input_dim, h_size=128, out=2, n_layers=4,
                                activation_function=nn.ReLU, dropout=0.3, lr=0.001, weight_decay=1e-4)
        # Charger le state_dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Mode évaluation
        self.encoder = encoder
        self.scaler = scaler
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features = [f for f in feature_names if f not in continuous_features]

    def __call__(self, X):
        """
        Make the model callable for DiCE compatibility.
        
        Args:
            X (pd.DataFrame or torch.Tensor): Input data.
        
        Returns:
            torch.Tensor: Model output (logits or probabilities).
        """
        if isinstance(X, pd.DataFrame):
            return torch.tensor(self.predict_proba(X), dtype=torch.float32)
        return self.model(X, apply_softmax=True)

    def predict_proba(self, X):
        """
        Prédire les probabilités pour DiCE.
        
        Args:
            X (pd.DataFrame): Données d'entrée.
        
        Returns:
            np.ndarray: Probabilités des classes.
        """
        # Prétraitement des données
        X_processed = X.copy()
        # Encoder les caractéristiques catégoriques (si applicable)
        for col in self.categorical_features:
            if col in X_processed.columns and col in self.encoder:
                X_processed[col] = self.encoder[col].transform(X_processed[col])
        # Scaler les caractéristiques continues
        if self.continuous_features:
            X_processed[self.continuous_features] = self.scaler.transform(X_processed[self.continuous_features])
        # Convertir en tenseur PyTorch
        X_tensor = torch.tensor(X_processed[self.feature_names].values, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_tensor, apply_softmax=True).numpy()
        return probs

    def predict(self, X):
        """
        Prédire les classes pour DiCE.
        
        Args:
            X (pd.DataFrame): Données d'entrée.
        
        Returns:
            np.ndarray: Classes prédites.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Paramètres configurables
csv_file_path = "diabetes.csv"  # Chemin du fichier CSV
target_column = "Outcome"  # Nom de la colonne cible
continuous_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
model_path = "diabetes_model.pt"  # Chemin du fichier .pt
encoder_path = "diabetes_model_encoders.pkl"  # Chemin du fichier .pkl pour l'encodeur
scaler_path = "diabetes_model_scaler.pkl"  # Chemin du fichier .pkl pour le scaler

csv_file_path = "breast_cancer.csv"
target_column = "irradiat"
continuous_features = ['deg-malig']
model_path = "breast_cancer_model.pt"
encoder_path = "breast_cancer_model_encoders.pkl"
scaler_path = "breast_cancer_model_scaler.pkl"

# Charger le dataset
dataset = load_custom_dataset(csv_file_path, target_column)
target = dataset[target_column]
feature_names = [col for col in dataset.columns if col != target_column]

# Charger les encodeurs et scalers
encoder, scaler = load_preprocessors(encoder_path, scaler_path)

# Diviser le dataset
train_dataset, test_dataset, _, _ = train_test_split(dataset,
                                                     target,
                                                     test_size=0.2,
                                                     random_state=0,
                                                     stratify=target)

# Préparer les données pour DiCE
d = dice_ml.Data(dataframe=train_dataset,
                 continuous_features=continuous_features,
                 outcome_name=target_column)

# Charger le modèle PyTorch et enveloppeur
model_wrapper = PyTorchModelWrapper(model_path, encoder, scaler, feature_names, continuous_features)
m = dice_ml.Model(model=model_wrapper, backend='PYT')

# Créer une instance DiCE
exp = dice_ml.Dice(d, m)

# Générer un contre-exemple pour les 100 premières instances du jeu de test
test_instances = test_dataset.drop(columns=target_column).iloc[:100]
cfes = []
distances = []
sparsities = []

for idx, row in test_instances.iterrows():
    query_instance = row.to_frame().T  # DataFrame d'une seule ligne
    try:
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=1, desired_class="opposite")
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        if cf_df.empty:
            print(f"Instance {idx+1}: Aucun contre-exemple généré.")
            continue
        cf_instance = cf_df.drop(columns=target_column).iloc[0].to_numpy()
        query_instance_np = query_instance.to_numpy().flatten()
        dist = calculate_distance(query_instance_np, cf_instance)
        sparse = sparsity(query_instance_np, cf_instance)
        distances.append(dist)
        sparsities.append(sparse)
        cfes.append(cf_df)
        print(f"Instance {idx+1}: Distance de Manhattan = {dist:.4f}, Sparsity = {sparse}")
    except Exception as e:
        print(f"Instance {idx+1}: Erreur lors de la génération du contre-exemple: {e}")

# Statistiques globales
total_instances = len(test_instances)
if distances:
    avg_distance = np.mean(distances)
    avg_sparsity = np.mean(sparsities)
    success_ratio = len(cfes) / total_instances
    print("\nRésultats globaux :")
    print(f"Distance moyenne : {avg_distance:.4f}")
    print(f"Sparsity moyenne : {avg_sparsity:.4f}")
    print(f"Ratio de contre-exemples trouvés : {success_ratio:.4f} ({len(cfes)}/{total_instances})")
else:
    print("Aucun contre-exemple généré pour les 100 premières instances.")
    print(f"Ratio de contre-exemples trouvés : 0.0000 (0/{total_instances})")

# Visualiser les contre-exemples du premier exemple (optionnel)
# if cfes:
#     cfes[0].visualize_as_dataframe(show_only_changes=True)