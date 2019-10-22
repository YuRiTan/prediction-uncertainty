import os
import torch
from torch import nn


class Dropout_on_dim(torch.nn.modules.dropout._DropoutNd):
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
    

class DeepQuantileRegression(nn.Module):
    """ Monte Carlo Dropout Quantile Regression Neural Network """
    def __init__(self, params):
        super().__init__()
        self.hidden_size = params['hidden_size']
        self.quantiles = params['quantiles']
        self.model_ = nn.Sequential(
            nn.Linear(params['input_size'], params['hidden_size']),
#             nn.ReLU(),  # when you want to model non-linearities, but not
            Dropout_on_dim(params['dropout_p'], dim=params['dropout_dim']),
            nn.Linear(params['hidden_size'], len(params['quantiles']))
        )
        self.optim_ = torch.optim.Adam(
            self.model_.parameters(), 
            lr=params['lr'], 
            weight_decay=params['weight_decay']
        )
    
    def forward(self, X):
        return self.model_(X)
    
    def mc_predict(self, X, samples=4000):
        with torch.no_grad():
            self.model_.train()
            preds = torch.stack([self.model_(X) for _ in range(samples)], dim=-1)
        return preds
    
    def save(self, path, fname):
        torch.save({
            "model_state_dict": self.model_.state_dict(),
            "optim_state_dict": self.optim_.state_dict(),
        }, os.path.join(path, fname))
    
    def load(self, path, fname):
        checkpoint = torch.load(os.path.join(path, fname))
        self.model_.load_state_dict(checkpoint['model_state_dict'])
        self.optim_.load_state_dict(checkpoint['optim_state_dict'])
        return self