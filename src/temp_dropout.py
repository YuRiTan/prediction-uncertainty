import torch
from torch import nn


class MC_Dropout_Layer(nn.Module):
    """ https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/notebooks/regression/mc_dropout_homo.ipynb """
    def __init__(self, input_dim, output_dim, dropout_prob):
        super(MC_Dropout_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        
        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        
    def forward(self, x):
        dropout_mask = torch.bernoulli((1 - self.dropout_prob)*torch.ones(self.weights.shape))
        return torch.mm(x, self.weights*dropout_mask) + self.biases
