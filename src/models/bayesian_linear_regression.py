import logging

import theano
import pymc3 as pm
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from src.models.abstract_model import AbstractModel


logger = logging.getLogger(__name__)


class BayesianLinearRegression(AbstractModel):
    def __init__(self, params):
        self.params = params 
        # Dummy shared variables
        shared_x = theano.shared(np.zeros((1, self.params['input_size'])))
        shared_y = theano.shared(np.zeros((1, 1)))
        self.shared_vars = {'x': shared_x, 'y': shared_y}
        self.model = None
        self.trace = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def _set_shared_vars(self, values):
        for k, v in values.items():
            self.shared_vars[k].set_value(v)

    def _create_model(self):
        with pm.Model() as m:
            alpha = pm.Normal('alpha', mu=0.0, sd=1.0)
            beta = pm.Normal('beta', mu=0.0, sd=1.0, 
                             shape=self.params['input_size'])
            beta_x = pm.math.dot(self.shared_vars['x'], beta)
            mu = pm.Deterministic('mu', alpha + beta_x) 
            sd_scale = pm.Normal('sd_scale', mu=0.0, sd=1.0)
            sd_bias = pm.HalfNormal('sd_bias', sd=1.0) + 1e-5
            sd = pm.Deterministic('sigma', sd_bias + mu * sd_scale)
            obs = pm.Normal('obs', mu, sd=sd, observed=self.shared_vars['y'].T)
        return m

    def fit(self, x, y, **fit_kwargs):
        """ SKlearn `fit()`-like method to sample with input features `x`
        and observations `y` using `pm.sample`
        """
        x = self.feature_scaler.fit_transform(x)
        y = self.target_scaler.fit_transform(y)
        self._set_shared_vars({'x': x, 'y': y})
        self.model = self._create_model()
        self.trace = pm.sample(model=self.model, **fit_kwargs)
        return self

    def sample_posterior_predictive(self, x, **inference_kwargs):
        """ SKlearn `predict()`-like method to sample from the posterior 
        predictive using `pm.sample_ppc`.
        """
        if self.trace is None:
            raise NotFittedError("Please fit the model before predicting")
        if 'samples' not in inference_kwargs:
            inference_kwargs['samples'] = len(self.trace) * self.trace.nchains

        x = self.feature_scaler.transform(x)
        self._set_shared_vars({'x': x, 'y': np.zeros((x.shape[0], 1))})
        posterior = pm.sample_ppc(trace=self.trace, model=self.model, 
                                  **inference_kwargs)

        # return shape: batch_size x sample_size
        posterior = posterior['obs'].T[:, 0, :]
        return self.target_scaler.inverse_transform(posterior)

    def predict_quantiles(self, x, quantiles, **inference_kwargs):
        """ SKlearn `predict()`-like method to sample from the posterior 
        predictive using `pm.sample_ppc`.
        """
        pp_samples = self.sample_posterior_predictive(x, **inference_kwargs)
        # shape: batch_size x n_quantiles
        return np.quantile(pp_samples, quantiles, axis=1).T

    def plot_trace(self, skip_first_n=0):
        if self.trace is None:
            raise NotFittedError("Please fit the model before predicting")
        pm.traceplot(self.trace[skip_first_n:])

    def save(self, filepath, **save_kwargs):
        """ Saves trace of the PyMC3 model. """
        pm.save_trace(self.trace, directory=filepath, **save_kwargs)

    def load(self, filepath):
        """ recreates the model, and loads the trace of the PyMC3 model. """
        self.model = self._create_model()
        self.trace = pm.load_trace(filepath, model=self.model)
        return self
