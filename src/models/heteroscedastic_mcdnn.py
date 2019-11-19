import os
import torch
from torch import nn
import numpy as np


def gaussian_nll_loss(output, target, sigma, no_dim):
    sigma = torch.exp(sigma)

    # The manual way:
#     exp = -((target - output)**2) / (2 * sigma**2)
#     log_coef = -torch.log(sigma)
#     const = -0.5*np.log(2*np.pi)
#     loss = -(exp + log_coef + const).sum()

    # The automagic way
    dist = torch.distributions.Normal(output, sigma)  # or another distribution if preferred
    loss = -dist.log_prob(target)

    return loss.sum()


class HeteroscedasticDropoutNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hidden_size = params['hidden_size']
        self.model_ = nn.Sequential(
            nn.Linear(params['input_size'], params['hidden_size']),
            #  nn.PReLU(),  # when modelling non-linearities
            nn.Dropout(params['dropout_p']),
            nn.Linear(params['hidden_size'], params['output_size'])
        )
        self.optim_ = torch.optim.Adam(
            self.model_.parameters(), 
            lr=params['lr']
        )
    
    def forward(self, X):
        return self.model_(X)
        
    
    def mc_predict(self, X, samples=4000):
        with torch.no_grad():
            self.model_.train()
            preds = torch.stack([self.model_(X) for _ in range(samples)], dim=-1)
        return preds.permute(2, 0, 1)  # shape: n_samples, batch_size, output_dim 
    
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
