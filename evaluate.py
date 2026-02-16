"""
Оценка качества: MAE, RMSE.
"""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (ц/га)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error (ц/га)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Возвращает MAE и RMSE."""
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}
