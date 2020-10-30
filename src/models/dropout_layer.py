import torch
from torch import nn


class Dropout_on_dim(nn.modules.dropout._DropoutNd):
    """ Dropout that masks the (hidden) features equally over a batch """
    def __init__(self, p, dim=1, **kwargs):
        super().__init__(p, **kwargs)
        self.dropout_dim = dim
        self.multiplier_ = 1.0 / (1.0-self.p)  # normalize for dropout rate
        
    def forward(self, X):
        mask = torch.bernoulli(
                X.new(X.size(self.dropout_dim)).fill_(1-self.p)
        )
        return X * mask * self.multiplier_