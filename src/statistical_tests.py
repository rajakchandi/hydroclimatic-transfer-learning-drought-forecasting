import numpy as np
from scipy.stats import norm


def diebold_mariano(y_true, pred_a, pred_b, horizon=1):
    """
    Two-sided DM test using squared-error loss and a Bartlett/Newey-West
    long-run variance estimate. For overlapping h-step forecasts, use at
    least horizon-1 lags. Interpret cautiously with short test samples.
    """
    e_a = np.asarray(y_true) - np.asarray(pred_a)
    e_b = np.asarray(y_true) - np.asarray(pred_b)
    d = e_a**2 - e_b**2
    n = len(d)
    if n < 5:
        raise ValueError("DM test requires at least 5 paired forecast errors.")
    dbar = d.mean()
    q = max(int(horizon) - 1, 0)
    q = min(q, n - 1)
    centered = d - dbar
    lrv = np.dot(centered, centered) / n
    for lag in range(1, q + 1):
        gamma = np.dot(centered[lag:], centered[:-lag]) / n
        lrv += 2 * (1 - lag / (q + 1)) * gamma
    if lrv <= 0:
        return {"DM": np.nan, "p_value": np.nan, "mean_loss_diff": dbar}
    dm = dbar / np.sqrt(lrv / n)
    p = 2 * (1 - norm.cdf(abs(dm)))
    return {"DM": float(dm), "p_value": float(p), "mean_loss_diff": float(dbar)}
