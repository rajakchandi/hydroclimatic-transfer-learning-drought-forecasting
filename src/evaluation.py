import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - np.sum((y_true - y_pred) ** 2) / denom if denom > 0 else np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if abs(np.mean(y_true)) > 1e-12 else np.nan
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return {"R2": r2_score(y_true, y_pred), "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_true, y_pred), "NSE": nse, "KGE": kge}
