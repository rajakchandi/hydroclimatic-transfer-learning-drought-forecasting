import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import config
from src.data_preprocessing import read_domain_csv, chronological_windows, make_loader
from src.models import build_model
from src.training import train_model, predict
from src.transfer_learning import pretrain_source, adapt_target
from src.evaluation import regression_metrics
from src.visualization import plot_observed_predicted


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chronological_loader_split(X, y, batch_size):
    n = len(X)
    n_val = max(1, int(n * config.VALIDATION_FRACTION))
    train_loader = make_loader(X[:-n_val], y[:-n_val], batch_size, shuffle=True)
    val_loader = make_loader(X[-n_val:], y[-n_val:], batch_size, shuffle=False)
    return train_loader, val_loader


def run():
    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE if config.DEVICE == "cpu" or torch.cuda.is_available() else "cpu"
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    src = read_domain_csv(config.SOURCE_CSV, config.DATE_COLUMN,
                          config.FEATURE_COLUMNS, config.TARGET_COLUMN)
    tgt = read_domain_csv(config.TARGET_CSV, config.DATE_COLUMN,
                          config.FEATURE_COLUMNS, config.TARGET_COLUMN)

    all_rows = []
    for lead in config.LEAD_TIMES:
        # Fit source scalers on source data; apply the same transforms to target.
        Xs, ys, _, sx, sy = chronological_windows(
            src, config.FEATURE_COLUMNS, config.TARGET_COLUMN,
            config.SEQUENCE_LENGTH, lead, fit_scalers=True)
        Xt, yt, tidx, _, _ = chronological_windows(
            tgt, config.FEATURE_COLUMNS, config.TARGET_COLUMN,
            config.SEQUENCE_LENGTH, lead, x_scaler=sx, y_scaler=sy)

        # Hold out the final target segment before adaptation. No test data is used
        # for fitting, tuning or early stopping.
        cut = int(len(Xt) * (1 - config.TEST_FRACTION))
        Xdev, ydev = Xt[:cut], yt[:cut]
        Xtest, ytest = Xt[cut:], yt[cut:]
        test_loader = make_loader(Xtest, ytest, config.BATCH_SIZE, shuffle=False)
        src_train, src_val = chronological_loader_split(Xs, ys, config.BATCH_SIZE)

        for fraction in config.TARGET_ADAPTATION_FRACTIONS:
            n_adapt = max(2, int(len(Xdev) * fraction))
            Xadapt, yadapt = Xdev[:n_adapt], ydev[:n_adapt]
            adapt_train, adapt_val = chronological_loader_split(
                Xadapt, yadapt, config.BATCH_SIZE)

            for model_name in ["CNN", "LSTM", "BiLSTM_Attention", "CNN_LSTM",
                               "CNN_BiLSTM_Attention"]:
                model = build_model(model_name, len(config.FEATURE_COLUMNS),
                                    config.CNN_CHANNELS, config.HIDDEN_SIZE,
                                    config.DROPOUT)
                kwargs = dict(epochs=config.EPOCHS, lr=config.LEARNING_RATE,
                              weight_decay=config.WEIGHT_DECAY,
                              patience=config.PATIENCE, device=device)
                if model_name == "CNN_BiLSTM_Attention":
                    pretrained, source_state, _ = pretrain_source(
                        model, src_train, src_val, **kwargs)
                    fitted, _ = adapt_target(
                        pretrained, source_state, adapt_train, adapt_val,
                        stage_epochs=min(20, config.EPOCHS),
                        l2_sp_lambda=config.L2_SP_LAMBDA, **kwargs)
                    tag = "CNN_BiLSTM_Attention_TL"
                else:
                    fitted, _ = train_model(model, adapt_train, adapt_val, **kwargs)
                    tag = model_name + "_NoTL"

                obs_scaled, pred_scaled = predict(fitted, test_loader, device)
                obs = sy.inverse_transform(obs_scaled.reshape(-1, 1)).ravel()
                pred = sy.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
                metrics = regression_metrics(obs, pred)
                row = {"lead": lead, "adaptation_fraction": fraction,
                       "model": tag, **metrics}
                all_rows.append(row)
                plot_observed_predicted(
                    obs, pred,
                    config.FIGURE_DIR / f"{tag}_lead{lead}_adapt{fraction:.2f}.png",
                    title=f"{tag} | lead={lead} | adaptation={fraction:.0%}")
                torch.save(fitted.state_dict(),
                           config.MODEL_DIR / f"{tag}_lead{lead}_adapt{fraction:.2f}.pt")

    results = pd.DataFrame(all_rows)
    results.to_csv(config.TABLE_DIR / "model_performance.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    run()
