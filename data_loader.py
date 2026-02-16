"""
Загрузка данных: NASA POWER (метео) + Eurostat (урожайность)
Для демо использует сгенерированные данные, если API недоступен.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def fetch_nasa_power(lat: float = 50.0, lon: float = 10.0, start: str = "2015-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """
    Загрузка метеоданных через NASA POWER API.
    Параметры: температура, осадки, радиация.
    """
    if not REQUESTS_AVAILABLE:
        return _generate_weather_data(start, end)
    
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_PAR_TOT",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"API error: {resp.status_code}")
        
        data = resp.json()
        props = data.get("properties", {}).get("parameter", {})
        
        df = pd.DataFrame(props)
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "T2M": "temperature",
            "PRECTOTCORR": "precipitation", 
            "ALLSKY_SFC_PAR_TOT": "solar_radiation"
        })
        # -999 = пропуск
        df = df.replace(-999, np.nan).ffill().bfill()
        return df.resample("ME").mean()  # месячные средние
    except Exception as e:
        warnings.warn(f"NASA POWER недоступен ({e}), используем синтетические данные")
        return _generate_weather_data(start, end)


def _generate_weather_data(start: str, end: str) -> pd.DataFrame:
    """Синтетические метеоданные для демо."""
    dates = pd.date_range(start=start, end=end, freq="ME")
    np.random.seed(42)
    n = len(dates)
    temp = 15 + 10 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 2
    precip = 50 + 30 * np.random.rand(n)
    solar = 15 + 5 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 2
    return pd.DataFrame({
        "temperature": np.clip(temp, -5, 35),
        "precipitation": np.clip(precip, 0, 200),
        "solar_radiation": np.clip(solar, 5, 25)
    }, index=dates)


def fetch_ndvi_evi_synthetic(regions: list, start: str, end: str) -> pd.DataFrame:
    """
    Синтетические спутниковые индексы NDVI, EVI.
    В реальном проекте: Google Earth Engine, Sentinel Hub и т.д.
    """
    dates = pd.date_range(start=start, end=end, freq="ME")
    np.random.seed(43)
    rows = []
    for region in regions:
        base_ndvi = 0.5 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, len(dates)))
        base_evi = 0.3 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, len(dates)))
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "region": region,
                "NDVI": np.clip(base_ndvi[i] + np.random.randn() * 0.05, 0.1, 0.9),
                "EVI": np.clip(base_evi[i] + np.random.randn() * 0.03, 0.1, 0.6)
            })
    return pd.DataFrame(rows)


def fetch_eurostat_yield(crop: str = "wheat") -> pd.DataFrame:
    """
    Eurostat crop yield (ц/га). Для демо — синтетические данные.
    Реальный источник: https://ec.europa.eu/eurostat
    """
    regions = ["DE", "FR", "PL", "ES", "IT"]
    years = list(range(2015, 2024))
    np.random.seed(44)
    rows = []
    for region in regions:
        base = 50 + np.random.randint(-10, 15)
        for y in years:
            yield_val = base + (y - 2019) * 2 + np.random.randn() * 5
            rows.append({"region": region, "year": y, "yield_tha": max(20, min(80, yield_val))})
    return pd.DataFrame(rows)


def build_dataset(regions: list = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Собирает объединённый датасет: метео + NDVI/EVI + урожайность.
    Возвращает (X, y) для обучения.
    """
    regions = regions or ["DE", "FR", "PL"]
    start, end = "2015-01-01", "2023-12-31"
    
    # Метео (одна точка — упрощение)
    weather = fetch_nasa_power(start=start, end=end)
    weather = weather.reset_index().rename(columns={"index": "date"})
    weather["date"] = pd.to_datetime(weather["date"])
    weather["year"] = weather["date"].dt.year
    weather["month"] = weather["date"].dt.month
    
    # Спутниковые индексы
    sat = fetch_ndvi_evi_synthetic(regions, start, end)
    
    # Урожайность по регионам и годам
    yields_df = fetch_eurostat_yield()
    
    # Агрегация: усредняем метео и индексы по году для каждого региона
    rows = []
    for region in regions:
        reg_yields = yields_df[yields_df["region"] == region]
        reg_sat = sat[sat["region"] == region]
        for _, row in reg_yields.iterrows():
            y = int(row["year"])
            yield_val = row["yield_tha"]
            sat_year = reg_sat[reg_sat["date"].dt.year == y]
            weather_year = weather[weather["year"] == y]
            if len(sat_year) > 0 and len(weather_year) > 0:
                rows.append({
                    "region": region,
                    "year": y,
                    "yield_tha": yield_val,
                    "NDVI_mean": sat_year["NDVI"].mean(),
                    "EVI_mean": sat_year["EVI"].mean(),
                    "temp_mean": weather_year["temperature"].mean(),
                    "precip_sum": weather_year["precipitation"].sum(),
                    "solar_mean": weather_year["solar_radiation"].mean()
                })
    
    df = pd.DataFrame(rows)
    X = df[["NDVI_mean", "EVI_mean", "temp_mean", "precip_sum", "solar_mean"]]
    y = df["yield_tha"]
    return df, X, y
