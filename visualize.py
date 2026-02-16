"""
Визуализация прогноза по регионам.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions_by_region(df: pd.DataFrame, region_preds: dict):
    """
    График: фактическая vs прогнозная урожайность по регионам.
    region_preds: {region: (years, actual, predicted)}
    """
    regions = list(region_preds.keys())
    n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        years, actual, pred = region_preds[region]
        ax.plot(years, actual, "o-", label="Факт", color="steelblue")
        ax.plot(years, pred, "s--", label="Прогноз", color="coral")
        ax.set_title(f"Регион {region}")
        ax.set_xlabel("Год")
        ax.set_ylabel("Урожайность (ц/га)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("predictions_by_region.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Сохранено: predictions_by_region.png")


def plot_fact_vs_pred(y_true: np.ndarray, y_pred: np.ndarray):
    """График: факт vs прогноз (scatter)."""
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "k--", label="Идеальный прогноз")
    plt.xlabel("Фактическая урожайность (ц/га)")
    plt.ylabel("Прогнозная урожайность (ц/га)")
    plt.title("Факт vs Прогноз")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("fact_vs_pred.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Сохранено: fact_vs_pred.png")
