"""
LSTM модель для прогноза урожайности по временным рядам.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class CropYieldLSTM(nn.Module):
    """LSTM для регрессии урожайности (ц/га)."""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # последний шаг


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 3) -> tuple:
    """Создаёт последовательности для LSTM из yearly данных."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def train_model(X: np.ndarray, y: np.ndarray, seq_len: int = 3, epochs: int = 100) -> tuple:
    """
    Обучает LSTM и возвращает модель, scaler, метрики.
    """
    X_seq, y_seq = create_sequences(X, y, seq_len)
    if len(X_seq) < 5:
        raise ValueError("Недостаточно данных для создания последовательностей. Добавьте больше регионов/лет.")
    
    scaler = StandardScaler()
    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    scaler.fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(X_seq.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_seq, test_size=0.2, random_state=42)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    
    model = CropYieldLSTM(input_size=X.shape[1], hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()
    
    return model, scaler, y_test, y_pred


def predict_by_region(df: pd.DataFrame, model, scaler, seq_len: int = 3) -> dict:
    """
    Прогноз по регионам: для каждого региона последние seq_len лет -> предсказание.
    Возвращает {region: (years, actual, predicted)}.
    """
    feat_cols = ["NDVI_mean", "EVI_mean", "temp_mean", "precip_sum", "solar_mean"]
    region_preds = {}
    for region in df["region"].unique():
        reg_df = df[df["region"] == region].sort_values("year").reset_index(drop=True)
        if len(reg_df) < seq_len + 1:
            continue
        X = reg_df[feat_cols].values.astype(np.float32)
        y = reg_df["yield_tha"].values
        X_seq, y_seq = create_sequences(X, y, seq_len)
        X_flat = X_seq.reshape(-1, X_seq.shape[-1])
        X_scaled = scaler.transform(X_flat).reshape(X_seq.shape)
        X_t = torch.FloatTensor(X_scaled)
        model.eval()
        with torch.no_grad():
            pred = model(X_t).numpy().flatten()
        years = reg_df["year"].values[seq_len:]
        region_preds[region] = (years, y_seq, pred)
    return region_preds
