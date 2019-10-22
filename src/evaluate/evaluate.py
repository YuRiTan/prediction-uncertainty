import numpy as np
from scipy.integrate import simps, quad
from scipy.interpolate import interp1d


def crps_riemann(y_true, y_pred, qs):
    def _single_crps(y_true, y_pred, qs):
        mask = np.where(y_pred > y_true, True, False)
        int_cdf = interp1d(y_pred, qs, fill_value='extrapolate')
        int_q = int_cdf(y_true)
        y_left = np.append(y_pred[~mask], y_true)
        y_right = np.append(y_true, y_pred[mask])
        q_left = np.append(QUANTILES[~mask], int_q)
        q_right = np.append(int_q, QUANTILES[mask])
        return simps(q_left, y_left) + simps(1 - q_right, y_right)
        
    crps_ = np.array([_single_crps(y_t, y_p, qs)
                      for y_t, y_p in zip(y_true, y_pred)])
    return np.mean(crps_)


def crps_single(x, cdf, xmin=None, xmax=None):
    if xmin is None or xmax is None:
        xmin, xmax = np.min(cdf.x), np.max(cdf.x)
        
    lhs = lambda y: np.square(cdf(y))
    rhs = lambda y: np.square(1. - cdf(y))
    
    lhs_int, lhs_tol = quad(lhs, xmin, x)
    rhs_int, rhs_tol = quad(rhs, x, xmax)

    return lhs_int + rhs_int

crps = np.vectorize(crps_single)


def spread(preds):
    """ Calculates average distance of the quantile predictions
    to each other over all quantiles """
    spreads = np.zeros(preds.shape[0])
    avg_factor = 2 * preds.shape[1] * (preds.shape[1] - 1)
    for i in range(preds.shape[0]):
        spreads[i] = (
            np.sum(np.abs(np.subtract.outer(preds[i], preds[i]))) / avg_factor
        )
    return spreads


def distance(y_true, preds):
    """ Calculates the average distance of the quantile predictions
    compared to the actual value """
    return np.mean(np.abs(preds - y_true.reshape(-1, 1)), axis=1)


def rps(y_true, preds):
    """ Function that calculates the Ranked Probability Score """
    rps_scores = distance(y_true, preds) - spread(preds)
    return np.mean(rps_scores)