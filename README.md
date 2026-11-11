# Hydroclimatic Transfer Learning for Drought Forecasting

Modular PyTorch implementation scaffold for monthly SPEI forecasting using source-to-target transfer learning.

## Project layout
- `config.py`: data paths, predictors, target, lead times and training settings.
- `src/data_preprocessing.py`: chronological split, scaling and sliding windows.
- `src/models.py`: CNN, LSTM, BiLSTM-Attention, CNN-LSTM and CNN-BiLSTM-Attention.
- `src/training.py`: supervised training, early stopping and optional Optuna tuning.
- `src/transfer_learning.py`: source pretraining, staged target adaptation and L2-SP.
- `src/evaluation.py`: R², RMSE, MAE, NSE and KGE.
- `src/statistical_tests.py`: Diebold–Mariano test with HAC variance estimate.
- `src/shap_analysis.py`: optional SHAP analysis for sequence inputs.
- `src/visualization.py`: observed-versus-predicted and feature-importance plots.
- `main.py`: experiment entry point.

## Data format
Provide one CSV per domain (source and target). Each CSV should contain:
- one monthly date column (default: `Date`);
- the predictor columns listed in `config.py`;
- the target column (default: `SPEI_6`).

Rows must be monthly and chronologically ordered. Do not include future information in predictors.

## Run
1. Install packages: `pip install -r requirements.txt`
2. Edit `config.py` with your actual paths and column names.
3. Run: `python main.py`

This is a modular implementation template, not a verbatim conversion of your complete original notebook. Model architecture and experiment interfaces are included; check the data format and methodological choices against your manuscript before reporting results. In particular, define the target-domain adaptation/test split without using test observations for tuning, and fit scalers only on the relevant training data.
