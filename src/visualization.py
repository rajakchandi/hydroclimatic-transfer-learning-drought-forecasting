from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_observed_predicted(y_true, y_pred, output_path, title="Observed vs predicted"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, alpha=0.65)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax.set(xlabel="Observed", ylabel="Predicted", title=title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_shap_importance(shap_values, feature_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance = np.mean(np.abs(shap_values), axis=(0, 1))
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.asarray(feature_names)[order], importance[order])
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
