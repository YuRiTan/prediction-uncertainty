import matplotlib.pyplot as plt

from src.visualization import beautify_ax


def plot_quantile_calibration(target, preds, quantiles):
    assert preds.shape[0] == len(target)
    assert preds.shape[1] == len(quantiles)
    
    quantile_scores = [
        (preds[:, q] >= target).sum() / len(preds)
        for q in range(len(quantiles))
    ]
    
    plt.plot(quantiles, quantile_scores, label='predictions')
    plt.plot(quantiles, quantiles, label='perfectly calibrated')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('Calibration plot')
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax = beautify_ax(ax)
    return ax
