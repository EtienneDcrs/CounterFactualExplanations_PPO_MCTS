# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:19:32 2020

@author: Iacopo
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from CERTIFAI import CERTIFAI
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping
import os

patience = 10
max_epochs = 200

early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    strict = False,
    verbose = False,
    mode = 'min')

class Classifier(pl.LightningModule):
    def __init__(self, in_feats = 5, h_size = 25, out = 5, n_layers = 1,
                 activation_function = nn.ReLU, lr = 1e-3):
        super().__init__()
        
        mlp_sequence = OrderedDict([('mlp1', nn.Linear(in_features = in_feats,
                                                       out_features = h_size)),
                                    ('activ1', activation_function())])
        
        for i in range(1,n_layers):
            new_keys = ['mlp'+str(i+1), 'activ'+str(i+1)]
            mlp_sequence[new_keys[0]] = nn.Linear(in_features = h_size,
                                             out_features = h_size)
            mlp_sequence[new_keys[1]] = activation_function()
        
        mlp_sequence['out_projection'] = nn.Linear(in_features = h_size,
                                                   out_features = out)
        
        self.net = nn.Sequential(mlp_sequence)
        
        self.learning_rate = lr
    
    def forward(self, x, apply_softmax = False):
        y = self.net(x)
        if apply_softmax:
            return nn.functional.softmax(y, -1)
        return y
    
    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr = self.learning_rate)
        
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
    
url = 'drug200.csv'

cert = CERTIFAI.from_csv(url)

model_input = cert.transform_x_2_input(cert.tab_dataset, pytorch = True)

target = model_input[:,-1].long()

cert.tab_dataset = cert.tab_dataset.iloc[:,:-1]

predictors = model_input[:,:-1]

val_percentage = 0.1

batch_size = 8

train_loader = DataLoader(
    TensorDataset(predictors[:-int(len(predictors)*val_percentage)],
                  target[:-int(len(predictors)*val_percentage)]),
    batch_size = batch_size)

val_loader = DataLoader(
    TensorDataset(predictors[-int(len(predictors)*val_percentage):],
                  target[int(-len(predictors)*val_percentage):]),
    batch_size = batch_size)

# Check if the model file exists
model_path = 'model.pth'

if os.path.exists(model_path):
    print("Loading existing model...")
    model = Classifier()
    model.load_state_dict(torch.load(model_path))
else:
    print("Training new model...")
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop])
    model = Classifier()
    trainer.fit(model, train_loader, val_loader)  # Train the model
    # Save the trained model
    torch.save(model.state_dict(), model_path)

# Make predictions on the test set
test_loader = DataLoader(
    TensorDataset(predictors[-int(len(predictors) * val_percentage):],
                  target[int(-len(predictors) * val_percentage):]),
    batch_size=batch_size
)

# save the model
torch.save(model.state_dict(), 'model.pth')


import pandas as pd
import numpy as np

# Create a dataframe from your test data
test_data = """Age,Sex,BP,Cholesterol,Na_to_K
23,F,HIGH,HIGH,25.355
47,M,LOW,HIGH,13.093
47,M,LOW,HIGH,10.114
28,F,NORMAL,HIGH,7.79
61,F,LOW,HIGH,18.043
22,F,NORMAL,HIGH,8.60
49,F,NORMAL,HIGH,16.2
41,M,LOW,HIGH,11.037
60,M,NORMAL,HIGH,15.1
54,F,NORMAL,HIGH,15.361"""

# Save test data to a temporary file
with open('test_data.csv', 'w') as f:
    f.write(test_data)

# Load test data
test_df = pd.read_csv('test_data.csv')

# Transform test data using the same CERTIFAI transformation
test_input = cert.transform_x_2_input(test_df, pytorch=True)

# Create test loader
test_loader = DataLoader(
    TensorDataset(test_input), 
    batch_size=len(test_input)
)

# Set model to evaluation mode
model.eval()

# Run inference
with torch.no_grad():
    for batch in test_loader:
        x = batch[0]
        predictions = model(x, apply_softmax=True)
        
        # Get the predicted class (highest probability)
        predicted_classes = torch.argmax(predictions, dim=1)
        
        # Print results
        print("Test Results:")
        print("-" * 50)
        for i, (pred_class, pred_probs) in enumerate(zip(predicted_classes, predictions)):
            print(f"Sample {i+1}:")
            print(f"  Input: {test_df.iloc[i].to_dict()}")
            print(f"  Predicted Class: {pred_class.item()}")
            formatted_probs = [f"{prob * 100:.2f}%" for prob in pred_probs.numpy()]
            print(f"  Class Probabilities: {formatted_probs}")
            print("-" * 30)