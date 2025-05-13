from collections import OrderedDict
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
