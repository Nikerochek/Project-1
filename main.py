"""
Прогноз урожайности зерновых по спутниковым данным и метеоинформации.
Студенческий проект: LSTM, NASA POWER, Eurostat.
"""

from data_loader import build_dataset
from model import train_model, predict_by_region
from evaluate import evaluate
from visualize import plot_fact_vs_pred, plot_predictions_by_region


def main():
    print("Загрузка данных (метео + NDVI/EVI + урожайность)...")
    df, X, y = build_dataset(regions=["DE", "FR", "PL", "ES", "IT"])
    print(f"Датасет: {len(df)} записей, регионы: {df['region'].unique().tolist()}")

    print("\nОбучение LSTM...")
    model, scaler, y_test, y_pred = train_model(X.values, y.values, seq_len=3, epochs=150)

    metrics = evaluate(y_test, y_pred)
    print(f"\nМетрики на тестовой выборке:")
    print(f"  MAE:  {metrics['MAE']:.2f} ц/га")
    print(f"  RMSE: {metrics['RMSE']:.2f} ц/га")

    print("\nВизуализация...")
    plot_fact_vs_pred(y_test, y_pred)
    region_preds = predict_by_region(df, model, scaler, seq_len=3)
    plot_predictions_by_region(df, region_preds)

    print("\nГотово. Результаты сохранены в predictions_by_region.png и fact_vs_pred.png")


if __name__ == "__main__":
    main()
