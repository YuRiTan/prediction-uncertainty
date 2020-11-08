import os
import logging
import torch
import numpy as np

from scipy import stats
from functools import partial
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from src.evaluate.loss_functions import gaussian_nll_loss, QuantileLoss
from src.model.abstract_model import AbstractModel
from src.model.dropout_layer import Dropout_on_dim
from src.model.quantile_utils import get_quantile_pred, create_quantiles

logger = logging.getLogger(__name__)


class BaseMCDropoutNet(nn.Module, AbstractModel):
    def __init__(self):
        super().__init__()
        self.model_ = None
        self.optim_ = None
        self.criterion_ = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
    
    @staticmethod
    def _data_to_tensor(X, y=None):
        """ Assuming that we only have numeric features """
        is_tensor = lambda x: isinstance(x, torch.Tensor)
        X_ = X if is_tensor(X) else torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_ = y if is_tensor(y) else torch.tensor(y, dtype=torch.float32)
            return X_, y_
        else:
            return X_

    def fit(self, X, y, val_data=None, **fit_kwargs):
        X = self.feature_scaler.fit_transform(X)
        y = self.target_scaler.fit_transform(y)
        X, y = self._data_to_tensor(X, y)

        if val_data is not None:
            x_val = self.feature_scaler.transform(val_data[0])
            y_val = self.target_scaler.transform(val_data[1])
            x_val, y_val = self._data_to_tensor(x_val, y_val)

        train_dl = DataLoader(TensorDataset(X, y), 
                              fit_kwargs.get('batch_size', 32),
                              shuffle=fit_kwargs.get('shuffle', True))

        for epoch in range(fit_kwargs.get('epochs', 100)):
            for x_batch, y_batch in train_dl:
                self.model_.train()
                preds = self.model_(x_batch)
                loss = self.criterion_(preds, y_batch)
                loss.backward()
                self.optim_.step()
                self.optim_.zero_grad()

            if epoch % fit_kwargs.get('print_iter', 1) == 0:
                train_loss = self.evaluate(X, y)
                msg = f"Epoch: {epoch} \t Train loss:{train_loss:.5f}" 
                if val_data:
                    val_loss = self.evaluate(x_val, y_val)
                    msg += f"\t Val loss: {val_loss:.5f}"
                logger.debug(msg)

    def evaluate(self, X, y):
        X, y = self._data_to_tensor(X, y)
        self.model_.eval()
        with torch.no_grad():
            loss = self.criterion_(self.model_(X), y)

        return loss
    
    def save(self, filepath, **save_kwargs):
        torch.save({
            "model_state_dict": self.model_.state_dict(),
            "optim_state_dict": self.optim_.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.model_.load_state_dict(checkpoint['model_state_dict'])
        self.optim_.load_state_dict(checkpoint['optim_state_dict'])
        return self


class QuantileRegressionMCDropoutNet(BaseMCDropoutNet):
    """ Quantile Regression Monte Carlo Dropout Neural Network """
    def __init__(self, params):
        super().__init__()
        self.quantiles = params['quantiles']
        self.model_ = self._create_model(params)
        self.optim_ = torch.optim.Adam(self.model_.parameters(), 
                                       lr=params.get('lr', 1e-4))
        self.criterion_ = partial(QuantileLoss, quantiles=self.quantiles)

    @staticmethod
    def _create_model(params):
        return nn.Sequential(
            nn.Linear(params['input_size'], params['hidden_size']),
            Dropout_on_dim(params['dropout_p'], dim=params['dropout_dim']),
            nn.Linear(params['hidden_size'], len(params['quantiles']))
        )
    
    def forward(self, X):
        return self.model_(X)

    def mc_predict(self, X, samples=4000):
        X = self.feature_scaler.transform(X)
        X = self._data_to_tensor(X)
        with torch.no_grad():
            self.model_.train()
            preds = torch.stack([self.model_(X) for _ in range(samples)], dim=0)

        # shape: n_samples, batch_size, output_dim 
        preds = preds.numpy()
        return np.stack([self.target_scaler.inverse_transform(preds[:, :, q]) 
                         for q in range(preds.shape[-1])], axis=-1)

    def predict_quantiles(self, X, samples=1000):
        q_preds = self.mc_predict(X, samples)
        mean_q_preds = np.mean(q_preds, axis=0)
        return mean_q_preds

    def _quantiles_to_samples(self, preds):
        """ Mimic a continuous distribution by inter/extrapolating values 
        between the quantile predictions """
        return np.array([
            get_quantile_pred(q=stats.uniform.rvs(),  # random q to interpolate
                              used_quantiles=self.quantiles,
                              preds=preds[s, :])  # shape: samples, output_dim
            for s in range(preds.shape[0])  # sample from all samples
        ])

    def sample_posterior_predictive(self, X, samples=1000):
        """ Model definitely not designed for this functionality, but for 
        comparison purposes in blog.
        """
        preds = self.mc_predict(X, samples)
        samples = np.zeros((preds.shape[1], preds.shape[0]))
        for i in range(preds.shape[1]):
            samples[i] = self._quantiles_to_samples(preds[:, i, :])
        return samples
    

class HeteroscedasticMCDropoutNet(BaseMCDropoutNet):
    """ Fitting mu and sigma to parameterize a Normal distribution """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model_ = self._create_model(params)
        self.optim_ = torch.optim.Adam(self.model_.parameters(), 
                                       lr=params.get('lr', 1e-4))
        self.criterion_ = gaussian_nll_loss

    @staticmethod 
    def _create_model(params):
        return nn.Sequential(
            nn.Linear(params['input_size'], params['hidden_size']),
            nn.Dropout(params['dropout_p']),
            nn.Linear(params['hidden_size'], 2)
        )

    def forward(self, X):
        return self.model_(X) 

    def mc_predict(self, X, samples=1000):
        X = self.feature_scaler.transform(X)
        X = self._data_to_tensor(X)
        with torch.no_grad():
            self.model_.train()
            # transform to shape: n_samples, batch_size, output_dim 
            sampled_preds = [self.model_(X) for _ in range(samples)]
            preds = torch.stack(sampled_preds, dim=0).numpy()

        # Take the exp of sigma to make positive. This also happens in the loss
        # function. Unfortunately applying this in the forward method doesn't
        # work...
        preds[:, :, 1] = np.exp(preds[:, :, 1])

        inv_preds = np.stack([
            self.target_scaler.inverse_transform(preds[:, :, 0]),
            # Don't inverse_transform sigma! only use `scale_` parameter!
            preds[:, :, 1] / self.target_scaler.scale_
        ], axis=-1)

        return inv_preds

    def sample_posterior_predictive(self, X, samples=1000):
        preds = self.mc_predict(X, samples)
        mus = np.mean(preds[:, :, 0], axis=0)
        sigmas = np.mean(preds[:, :, 1], axis=0)
        spp = np.stack([
            stats.norm.rvs(size=samples, loc=mu, scale=sigma)
            for mu, sigma in zip(mus, sigmas)
        ], axis=0)
        # shape: batch_size x n_samples
        return spp
    
    def predict_quantiles(self, X, quantiles=None, samples=1000):
        """ Model not designed for this functionality, but for comparison 
        purposes in blog.
        """
        default_qs = self.params.get(
            'quantiles', create_quantiles(11, min_q=0.05, max_q=0.95)
        )
        quantiles = quantiles if quantiles is not None else default_qs
        sample_preds = self.sample_posterior_predictive(X, samples)
        # shape: batch_size x n_quantiles
        return np.quantile(sample_preds, quantiles, axis=1).T
