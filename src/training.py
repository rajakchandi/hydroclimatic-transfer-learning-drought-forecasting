import copy
import numpy as np
import torch
from torch import nn


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3,
                weight_decay=1e-4, patience=12, device="cpu",
                source_state=None, l2_sp_lambda=0.0):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    source_params = {}
    if source_state is not None:
        source_params = {k: v.to(device) for k, v in source_state.items()
                         if k in dict(model.named_parameters())}
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if source_params and l2_sp_lambda:
                penalty = 0.0
                for name, param in model.named_parameters():
                    if name in source_params and param.requires_grad:
                        penalty = penalty + torch.sum((param - source_params[name]) ** 2)
                loss = loss + l2_sp_lambda * penalty
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(loss_fn(model(xb.to(device)), yb.to(device)).item())
        v = float(np.mean(val_losses)) if val_losses else float(np.mean(train_losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)),
                        "val_loss": v})
        scheduler.step()
        if v < best_loss:
            best_loss, best_state, stale = v, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval()
    preds, obs = [], []
    for xb, yb in loader:
        preds.extend(model(xb.to(device)).cpu().numpy().ravel())
        obs.extend(yb.numpy().ravel())
    return np.asarray(obs), np.asarray(preds)
