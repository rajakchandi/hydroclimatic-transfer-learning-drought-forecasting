import copy
from .training import train_model


def pretrain_source(model, train_loader, val_loader, **training_kwargs):
    """Pretrain the complete network on source-domain training sequences."""
    model, history = train_model(model, train_loader, val_loader, **training_kwargs)
    return model, copy.deepcopy(model.state_dict()), history


def adapt_target(model, source_state, train_loader, val_loader,
                 stage_epochs=20, **training_kwargs):
    """
    Staged adaptation:
      1) freeze CNN/BiLSTM feature extractor; train attention/head;
      2) unfreeze all layers and fine-tune with L2-SP to source parameters.
    """
    model.load_state_dict(source_state)
    for name, p in model.named_parameters():
        p.requires_grad = not (name.startswith("cnn.") or name.startswith("bilstm."))
    model, hist_head = train_model(model, train_loader, val_loader,
                                   epochs=stage_epochs, **training_kwargs)

    for p in model.parameters():
        p.requires_grad = True
    model, hist_all = train_model(
        model, train_loader, val_loader, source_state=source_state,
        l2_sp_lambda=training_kwargs.pop("l2_sp_lambda", 1e-3),
        **training_kwargs)
    return model, {"head_stage": hist_head, "fine_tune_stage": hist_all}
