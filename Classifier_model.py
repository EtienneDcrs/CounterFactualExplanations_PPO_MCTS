from collections import OrderedDict
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping

class Classifier(pl.LightningModule):
    def __init__(self, in_feats=8, h_size=25, out=2, n_layers=1,
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



def train_model(dataset_path, model_path):
    # Load dataset to determine input features
    dataset = pd.read_csv(dataset_path)
    
    # Convert DataFrame to numpy array first, then to tensor
    model_input = torch.tensor(dataset.values, dtype=torch.float)
    
    # Extract predictors (all columns except the last one)
    predictors = model_input[:, :-1]
    
    # Extract target (the last column) - ensure it's long/int type for classification
    target = model_input[:, -1].long()

    val_percentage = 0.1
    batch_size = 8
    max_epochs = 200
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        strict=False,
        verbose=False,
        mode='min'
    )

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
    
    # Create model with correct input dimensions
    in_features = predictors.shape[1]
    num_classes = len(torch.unique(target))
    print(f"Creating model with {in_features} input features and {num_classes} output classes")
    
    model = Classifier(in_feats=in_features, out=num_classes)
    
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop])
    trainer.fit(model, train_loader, val_loader)
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Trained classifier model saved to {model_path}")
    
    return model