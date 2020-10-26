import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


def load_dataset(n=300, n_tst=150):
    """ Source: https://medium.com/tensorflow/regression-with-
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


def generate_sin_shaped_dataset(n=500, test_size=.2):
    """ Function that generates dummy data in a mirrored sine curved shape """
    def _parabolic_sine(x, amplitude=1):
        noise = amplitude * (np.random.rand(len(x))*2-1)
        return 8 + x**2/(32) + noise*8*np.sin(x/(4))

    x_min, x_max = 1, 30
    x = np.random.rand(n) * (x_max - x_min) 
    x, y = x.reshape(-1,1), _parabolic_sine(x, amplitude=1).reshape(-1,1)
    split_i = int(n * test_size)
    return x[:-split_i], x[-split_i:], y[:-split_i], y[-split_i:]


def load_boston_dataset(test_size=0.2, return_df=False):
    boston = load_boston()
    # first into df to make sure column names persist
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['target'] = boston.target 

    if return_df:
        return df
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            df.drop('target', axis=1), df.target, test_size=test_size
        )
        return (x_train.values, x_test.values,
                y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1))
