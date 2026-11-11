import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def read_domain_csv(path, date_col, feature_cols, target_col):
    df = pd.read_csv(path, parse_dates=[date_col])
    required = [date_col, *feature_cols, target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[required].sort_values(date_col).drop_duplicates(date_col)
    if df[date_col].isna().any():
        raise ValueError(f"{path}: invalid dates detected")
    return df.dropna(subset=[*feature_cols, target_col]).reset_index(drop=True)


def chronological_windows(df, feature_cols, target_col, sequence_length, lead,
                          x_scaler=None, y_scaler=None, fit_scalers=False):
    Xraw = df[feature_cols].to_numpy(dtype=np.float32)
    yraw = df[[target_col]].to_numpy(dtype=np.float32)
    if fit_scalers:
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        X = x_scaler.fit_transform(Xraw).astype(np.float32)
        y = y_scaler.fit_transform(yraw).astype(np.float32).ravel()
    else:
        if x_scaler is None or y_scaler is None:
            raise ValueError("Pass fitted scalers or set fit_scalers=True.")
        X = x_scaler.transform(Xraw).astype(np.float32)
        y = y_scaler.transform(yraw).astype(np.float32).ravel()

    xs, ys, target_indices = [], [], []
    # At origin t, use [t-sequence_length+1, ..., t] to predict t+lead.
    for t in range(sequence_length - 1, len(df) - lead):
        xs.append(X[t-sequence_length+1:t+1])
        ys.append(y[t+lead])
        target_indices.append(t+lead)
    return (np.asarray(xs, np.float32), np.asarray(ys, np.float32),
            np.asarray(target_indices), x_scaler, y_scaler)


def make_loader(X, y, batch_size=32, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).float().reshape(-1, 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
