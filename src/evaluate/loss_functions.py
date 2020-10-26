import torch

import logging

logger = logging.getLogger(__name__)


def gaussian_nll_loss(output, target):
    mu, sigma = output[:, :1], output[:, 1:]
    sigma = torch.exp(sigma)
    dist = torch.distributions.Normal(mu, sigma)
    loss = -dist.log_prob(target)
    
    return loss.sum()


def QuantileLoss(preds, target, quantiles):
    assert not target.requires_grad
    assert (preds.size(0) == target.size(0),
            f'preds.size:{preds.shape} target.size:{quantiles.shape}')
    assert (preds.size(1) == quantiles.shape[0], 
            f'preds.size:{preds.shape} quantiles.shape:{quantiles.shape}')

    def _tilted_loss(q, e):
        return torch.max((q-1) * e, q * e).unsqueeze(1)

    err = target - preds
    losses = [_tilted_loss(q, err[:, i])  # calculate per quantile
              for i, q in enumerate(quantiles)]

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss
