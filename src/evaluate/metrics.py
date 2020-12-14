import numpy as np


def mean_absolute_percentual_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
