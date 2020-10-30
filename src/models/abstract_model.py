class AbstractModel:
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict_quantiles(self, *args, **kwargs):
        raise NotImplementedError

    def sample_posterior_predictive(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError
