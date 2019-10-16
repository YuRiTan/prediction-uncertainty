import numpy as np

def load_dataset(n=300, n_tst=150):
    """ source: https://medium.com/tensorflow/regression-with-
    probabilistic-layers-in-tensorflow-probability-e46ff5d37baf 
    """
    w0, b0 = 0.125, 5.
    x_range = [-20, 60]
    
    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g**2.)
    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    x = x[..., np.newaxis]
    x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
    x_tst = x_tst[..., np.newaxis]
    return y, x, x_tst


def generate_dataset(n=500, test_size=.2):
    """ Method that generates dummy date with variance dependent on x """
    alpha, beta_0 = 2, 20
    x_min, x_max = 1, 4
    
    x = x_min + np.random.rand(n) * (x_max - x_min)
    # increasing variance with 2 minimum
    variance = np.random.randn(n) * np.where(abs(x * 5) < 2, 2, x*2)  
    y = (alpha * x  + beta_0) + variance
    # shift data to the positive
    x = x - x_min

    x, y = x.reshape(-1,1), y.reshape(-1,1)
    split_i = int(n * test_size)
    return x[:-split_i], x[-split_i:], y[:-split_i], y[-split_i:]
