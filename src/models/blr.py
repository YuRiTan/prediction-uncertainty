import theano
import pymc3 as pm
import numpy as np
from sklearn.exceptions import NotFittedError


class BayesianLinearRegression():
    def __init__(self):
        shared_x = theano.shared(np.zeros((1,1)))  # Dummy shared variable
        shared_y = theano.shared(np.zeros((1,1)))  # Dummy shared variable
        self.shared_vars = {'x': shared_x, 'y': shared_y}
        self.model = None
        self.trace = None

    def _set_shared_vars(self, values):
        for k,v in values.items():
            self.shared_vars[k].set_value(v)

    def _create_model(self):
        with pm.Model() as m:
            alpha = pm.Normal('alpha', 0, 10)
            beta = pm.Normal('beta', 0, 10)
            mu = pm.Deterministic('mu', alpha + beta * self.shared_vars['x'])
            sd_scale = pm.Normal('sd_scale', mu=0, sd=10)
            sd_bias = pm.HalfNormal('sd_bias', sd=10) + 1e-5
            sd = pm.Deterministic('sigma', sd_bias + mu * sd_scale)
            obs = pm.Normal('obs', mu, sd=sd, observed=self.shared_vars['y'])
        return m

    def fit(self, x, y, **fit_kwargs):
        """ SKlearn `fit()`-like method to sample with input features `x`
        and observations `y` using `pm.sample`
        """
        self._set_shared_vars({'x': x, 'y': y})
        self.model = self._create_model()
        self.trace = pm.sample(model=self.model, **fit_kwargs)
        return self

    def predict(self, x, **inference_kwargs):
        """ SKlearn `predict()`-like method to sample from the posterior predictive 
        with an out of sample test set using `pm.sample_ppc`.
        """
        if self.trace is None:
            raise NotFittedError("Please fit the model before predicting")
        if 'samples' not in inference_kwargs:
            inference_kwargs['samples'] = len(self.trace) * self.trace.nchains
        self._set_shared_vars({'x': x, 'y': np.zeros((x.shape[0], 1))})
        posterior = pm.sample_ppc(trace=self.trace, model=self.model, **inference_kwargs)
        return posterior['obs']


    def save_trace(self, fname, **save_kwargs):
        pm.save_trace(self.trace, directory=fname, **save_kwargs)

    def load_trace(self, fname):
        self.model = self._create_model()
        self.trace = pm.load_trace(fname, model=self.model)
        return self
