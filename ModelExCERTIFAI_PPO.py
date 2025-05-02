import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from CERTIFAI_PPO import CERTIFAI
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping

patience = 10
max_epochs = 15

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    strict=False,
    verbose=False,
    mode='min'
)

class Classifier(pl.LightningModule):
    def __init__(self, in_feats=5, h_size=25, out=5, n_layers=1,
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

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        self.log('test_loss', loss)
        
        # Calculate accuracy
        preds = torch.argmax(yhat, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_acc', acc)
        
        return {'test_loss': loss, 'test_acc': acc}


dataset = 'drug200'
dataset = 'adult'
dataset = 'drug5'

if dataset == 'adult':
    url = 'adult_with_headers_100.csv'
    model_path = 'adult_model.pt'
    out = 2
elif dataset == 'drug200' or dataset == 'drug5':
    url = dataset + '.csv'
    model_path = 'drug200_model.pt'
    out = 5

cert = CERTIFAI.from_csv(url)

# Prepare data
model_input = cert.transform_x_2_input(cert.tab_dataset, pytorch=True)
target = model_input[:, -1].long()
cert.tab_dataset = cert.tab_dataset.iloc[:, :-1]
predictors = model_input[:, :-1]
print(predictors.shape)

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





in_feats = predictors.shape[1]



# Check if model already exists
if os.path.exists(model_path):
    print("Loading existing model...")
    model = Classifier(in_feats=in_feats, out=out)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
        
else:
    print("Model not found, training a new one.")
    model = Classifier(in_feats=in_feats, out=out)

    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop])
    trainer.fit(model, train_loader, val_loader)

    # Save the model
    torch.save(model.state_dict(), model_path)


# Generate counterfactuals using PPO
cert.fit(model)

# print the results
# for result in cert.results:
#     print(result)

