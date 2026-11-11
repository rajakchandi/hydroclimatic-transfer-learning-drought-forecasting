import torch
from torch import nn


class CNNRegressor(nn.Module):
    def __init__(self, n_features, channels=32, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=3, padding=1),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(channels, 1))
    def forward(self, x):
        return self.head(self.features(x.transpose(1, 2)))


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.25, bidirectional=False):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True,
                           bidirectional=bidirectional)
        width = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))
    def forward(self, x):
        z, _ = self.rnn(x)
        return self.head(z[:, -1, :])


class AttentionPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                                   nn.Linear(width, 1))
    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class BiLSTMAttentionRegressor(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.25):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True,
                           bidirectional=True)
        self.attention = AttentionPool(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        z, _ = self.rnn(x)
        return self.head(self.attention(z))


class CNNLSTMRegressor(nn.Module):
    def __init__(self, n_features, channels=32, hidden=32, dropout=0.25):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_features, channels, 3, padding=1),
                                  nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.rnn(z)
        return self.head(z[:, -1, :])


class CNNBiLSTMAttentionRegressor(nn.Module):
    """CNN feature extractor + BiLSTM + temporal attention regression head."""
    def __init__(self, n_features, channels=32, hidden=32, dropout=0.25):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(n_features, channels, 3, padding=1),
                                  nn.GELU(), nn.Dropout(dropout))
        self.bilstm = nn.LSTM(channels, hidden, batch_first=True,
                              bidirectional=True)
        self.attention = AttentionPool(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.LayerNorm(64),
                                  nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, 32), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.bilstm(z)
        return self.head(self.attention(z))


def build_model(name, n_features, channels=32, hidden=32, dropout=0.25):
    models = {
        "CNN": lambda: CNNRegressor(n_features, channels, dropout),
        "LSTM": lambda: LSTMRegressor(n_features, hidden, dropout),
        "BiLSTM_Attention": lambda: BiLSTMAttentionRegressor(n_features, hidden, dropout),
        "CNN_LSTM": lambda: CNNLSTMRegressor(n_features, channels, hidden, dropout),
        "CNN_BiLSTM_Attention": lambda: CNNBiLSTMAttentionRegressor(
            n_features, channels, hidden, dropout),
    }
    if name not in models:
        raise ValueError(f"Unknown model {name}; choose from {list(models)}")
    return models[name]()
