import numpy as np


def explain_with_shap(model, background, samples, device="cpu", max_background=100):
    """
    Optional KernelExplainer wrapper. Flattens (sample,time,feature) arrays;
    returns SHAP values reshaped to the original sequence dimensions.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap to run explainability.") from exc
    import torch
    bg = background[:max_background]
    n_time, n_features = bg.shape[1], bg.shape[2]

    def predict_flat(z):
        z = np.asarray(z).reshape(-1, n_time, n_features).astype(np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(z).to(device)).cpu().numpy().ravel()
        return out

    explainer = shap.KernelExplainer(predict_flat, bg.reshape(len(bg), -1))
    values = explainer.shap_values(samples.reshape(len(samples), -1))
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[0]
    return values.reshape(len(samples), n_time, n_features)
