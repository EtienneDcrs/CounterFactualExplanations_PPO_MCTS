# -*- coding: utf-8 -*-

import os
import warnings
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from CERTIFAI import CERTIFAI 
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping

from Classifier_model import load_classifier_with_preprocessing
warnings.filterwarnings("ignore")



patience = 10
max_epochs = 200

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    strict=False,
    verbose=False,
    mode='min'
)

class Classifier(pl.LightningModule):
    def __init__(self, in_feats, h_size=25, out=2, n_layers=1,
                 activation_function=nn.ReLU, lr=1e-3):
        super().__init__()

        mlp_sequence = OrderedDict([('mlp1', nn.Linear(in_features=in_feats,
                                                       out_features=h_size)),
                                    ('activ1', activation_function())])

        for i in range(1, n_layers):
            new_keys = ['mlp' + str(i + 1), 'activ' + str(i + 1)]
            mlp_sequence[new_keys[0]] = nn.Linear(in_features=h_size,
                                                  out_features=h_size)
            mlp_sequence[new_keys[1]] = activation_function()

        mlp_sequence['out_projection'] = nn.Linear(in_features=h_size,
                                                   out_features=out)

        self.net = nn.Sequential(mlp_sequence)
        self.learning_rate = lr

    def forward(self, x, apply_softmax=False):
        y = self.net(x)
        if apply_softmax:
            return nn.functional.softmax(y, -1)
        return y

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimiser

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        self.log('val_loss', loss)

# Load dataset
url = 'data/diabetes_100.csv'
url = 'data/adult.csv'
cert = CERTIFAI.from_csv(url)

# Prepare data
model_input = cert.transform_x_2_input(cert.tab_dataset, pytorch=True)
target = model_input[:, -1].long()
cert.tab_dataset = cert.tab_dataset.iloc[:, :-1]
predictors = model_input[:, :-1]

val_percentage = 0.1
batch_size = 8

train_loader = DataLoader(
    TensorDataset(predictors[:-int(len(predictors) * val_percentage)],
                  target[:-int(len(predictors) * val_percentage)]),
    batch_size=batch_size
)

val_loader = DataLoader(
    TensorDataset(predictors[-int(len(predictors) * val_percentage):],
                  target[-int(len(predictors) * val_percentage):]),
    batch_size=batch_size
)

model_path = 'diabetes_model.pt'
model_path = 'adult_model.pt'
model_path = os.path.join("classification_models", model_path)
# Check if model already exists
# try:
#     model = Classifier(in_feats=predictors.shape[1])
#     model.load_state_dict(torch.load(model_path))
#     print("Model loaded successfully.")
# except FileNotFoundError:
#     print("Model not found, training a new one.")

#     # Train model
#     trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop])
#     model = Classifier()
#     trainer.fit(model, train_loader, val_loader)

#     # Save the model
#     torch.save(model.state_dict(), model_path)

model, label_encoders, scaler = load_classifier_with_preprocessing(url, model_path)


# Generate counterfactual
cert.fit(model, generations=15, verbose=True)

# Evaluate robustness and fairness
print("(Unnormalised) model's robustness:")
print(cert.check_robustness(), '\n')
print("(Normalised) model's robustness:")
print(cert.check_robustness(normalised=True), '\n')
print("Model's robustness for male subgroup vs. female subgroup (to check fairness of the model):")
print(cert.check_fairness([{'Sex': 'M'}, {'Sex': 'F'}]), '\n')
print("Visualising above results:")
print(cert.check_fairness([{'Sex': 'M'}, {'Sex': 'F'}], visualise_results=True), '\n')
print("Obtain feature importance in the model and visualise:")
print(cert.check_feature_importance(visualise_results=True))
