import numpy as np


def mean_absolute_percentual_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def ranked_probability_score(y_true, y_pred):
    raise NotImplementedError
