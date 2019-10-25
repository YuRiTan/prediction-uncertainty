import numpy as np
from scipy import interpolate
import torch


def create_quantiles(n, min_q=0.01, max_q=0.99):
    """ Create array of (evenly spaced) quantiles, given desired number of quantiles. """
    n -= 2  # because we add the lowest and highest quantiles manually
    n_equally_spaced = np.linspace(1 / (n + 1), 1 - 1 / (n + 1), n)
    quantiles = np.concatenate([np.array([min_q]), 
                                n_equally_spaced, 
                                np.array([max_q])])
    return quantiles


def get_quantile_pred(q, used_quantiles, preds):
    interp_cdf = interpolate.interp1d(used_quantiles, preds, fill_value='extrapolate')
    return interp_cdf(q)


def QuantileLoss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0), f'preds.size:{preds.shape} target.size:{quantiles.shape}'
    assert preds.size(1) == quantiles.shape[0], f'preds.size:{preds.shape} quantiles.shape:{quantiles.shape}'

    def _tilted_loss(q, e):
        return torch.max((q-1) * e, q * e).unsqueeze(1)

    err = target - preds
    losses = [_tilted_loss(q, err[:, i])  # calculate per quantile
              for i, q in enumerate(quantiles)]

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss 