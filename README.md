# Прогноз урожайности зерновых

Студенческий проект: регрессия урожайности (ц/га) по спутниковым индексам (NDVI, EVI) и метеоданным.

## Задача

- **Модель:** LSTM
- **Данные:** NASA POWER (метео), Eurostat (урожайность), спутниковые индексы NDVI/EVI
- **Метрики:** MAE, RMSE
- **Визуализация:** прогноз по регионам

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python main.py
```

Скрипт:
1. Загружает данные (при недоступности API использует синтетические данные)
2. Обучает LSTM
3. Выводит MAE и RMSE
4. Сохраняет графики: `fact_vs_pred.png`, `predictions_by_region.png`

## Структура

```
crop-yield-prediction/
├── main.py          # Точка входа
├── data_loader.py   # NASA POWER, Eurostat, NDVI/EVI
├── model.py         # LSTM
├── evaluate.py      # MAE, RMSE
├── visualize.py     # Визуализация по регионам
└── requirements.txt
```

## Источники данных

- **NASA POWER:** https://power.larc.nasa.gov/ — температура, осадки, радиация
- **Eurostat:** https://ec.europa.eu/eurostat — урожайность зерновых по странам
