# Weather Trend Forecasting

Forecasting global weather trends from the **Global Weather Repository** dataset, with a full data-science pipeline covering cleaning, EDA, time-series forecasting, anomaly detection, model ensembling, and advanced climate / air-quality / spatial / geographic analyses.

This project completes **both** the Basic and Advanced tracks of the assessment.

---

## Dataset

- **Source:** [World Weather Repository (Kaggle)](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
- **Size:** 145,590 daily observations × 41 features
- **Coverage:** 211 raw country labels (191 after de-duplication/normalization), 263 locations
- **Time span:** 2024-05-16 → 2026-06-05
- **Key features:** temperature, feels-like, humidity, precipitation, wind, pressure, cloud, UV index, visibility, air-quality (CO, ozone, NO₂, SO₂, PM2.5, PM10), and the `last_updated` timestamp used for time-series work.

Place the CSV at `data/GlobalWeatherRepository.csv` (already included).

---

## Project structure

```
weather-trend-forecasting/
├── README.md
├── requirements.txt
├── run_all.py                  # runs the whole pipeline in order
├── data/
│   └── GlobalWeatherRepository.csv
├── src/
│   ├── utils.py                # shared paths / IO helpers
│   ├── 01_data_cleaning.py     # cleaning, normalization, outliers, missing values
│   ├── 02_eda.py               # correlations + temperature/precipitation visuals
│   ├── 03_basic_forecast.py    # time-series forecast on last_updated (+ baseline)
│   ├── 04_anomaly_detection.py # Isolation Forest + z-score anomaly detection
│   ├── 05_ensemble_forecast.py # multi-model comparison + ensemble
│   └── 06_advanced_analyses.py # climate, air quality, feature importance, spatial, geographic
└── outputs/
    ├── metrics.json            # all model metrics (auto-generated)
    └── figures/                # all generated PNG visualizations
```

---

## How to run

```bash
# 1. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run everything
python run_all.py
```

Or run any stage individually, e.g. `python src/03_basic_forecast.py`.
Figures land in `outputs/figures/`; metrics in `outputs/metrics.json`.

> **Note on ARIMA:** the time-series stage uses scikit-learn (lag + seasonal features) so the repo installs with no compiler and is fully reproducible. The feature engineering is model-agnostic; if you install `statsmodels`, you can swap in ARIMA/SARIMA over the same daily series.

---

## Methodology & results

### 1. Data cleaning & preprocessing (`01_data_cleaning.py`)
- **Timestamp parsing** of `last_updated` and derived calendar features (year, month, day-of-year).
- **Country normalization:** the raw file mixes languages for the same country (e.g. `Inde`→India, `Bélgica`→Belgium, `火鸡`→Turkey). A mapping consolidates 211 labels into 191 canonical countries, each tagged with a **continent** for geographic analysis (0 unmapped).
- **Outliers:** IQR winsorization (1.5×IQR) on 16 numeric columns — ~125k extreme values capped rather than deleted to preserve sample size. Columns with a near-zero IQR (e.g. `visibility_km`, which is 10 km for most rows) are skipped so they aren't collapsed to a constant.
- **Missing values:** median impute (numeric) / mode impute (categorical). The source happened to be complete, but the pipeline is robust to gaps.
- **Normalization:** Min-Max scaled copies (`*_norm`) of modelling features.
- Output: `data/weather_clean.csv` (145,590 × 62).

### 2. Exploratory Data Analysis (`02_eda.py`)
- **Correlation heatmap** of core variables. Temperature correlates most with feels-like (r≈0.98), UV index (r≈0.50), and **inversely** with pressure (r≈−0.44) and humidity (r≈−0.35).
- **Temperature distribution:** mean ≈ 21.4 °C, range −2.4 → 46.0 °C.
- **Global daily trends** for temperature and precipitation over the two-year span.
- **Temperature by continent** boxplot.

Figures: `correlation_heatmap.png`, `temperature_distribution.png`, `global_temp_trend.png`, `global_precip_trend.png`, `temp_by_continent.png`.

### 3. Basic forecasting (`03_basic_forecast.py`)
Global mean daily temperature series built from `last_updated`. Lag features (t-1, t-2, t-3, t-7) + day-of-year seasonality, chronological 80/20 split.

| Model | MAE (°C) | RMSE (°C) | MAPE |
|---|---|---|---|
| Linear Regression | **0.43** | **1.26** | **4.0%** |
| Seasonal-Naive baseline | 0.65 | 1.49 | 5.2% |

The model clearly beats the naive baseline. Figure: `basic_forecast.png`.

### 4. Anomaly detection (`04_anomaly_detection.py`)
- **Isolation Forest** (multivariate, 7 features, 2% contamination) flags ~2,912 anomalies — extreme-heat / low-humidity desert readings (Abu Dhabi, Kuwait City, Baghdad at the 46 °C cap) dominate the top.
- **Z-score** (|z|>3) on temperature is included as a transparent univariate cross-check; it flags 0 here because winsorizing already removed univariate tails — a useful demonstration of why the multivariate method catches structure the simple rule misses.

Figures: `anomaly_scatter.png`, `anomaly_timeline.png`.

### 5. Forecasting with multiple models + ensemble (`05_ensemble_forecast.py`)
Extended features (lags up to 14, 7-day rolling mean, seasonality).

| Model | MAE (°C) | RMSE (°C) | MAPE |
|---|---|---|---|
| Linear Regression | 0.42 | 1.24 | 4.0% |
| **Random Forest** | 0.47 | **1.05** | 4.1% |
| Gradient Boosting | 0.57 | 1.10 | 4.6% |
| Ensemble (mean of 3) | 0.44 | 1.05 | **3.98%** |

**Random Forest** wins on RMSE; the **ensemble** ties on RMSE with the best MAPE, showing averaging stabilizes error. Figures: `ensemble_forecast.png`, `model_comparison.png`.

### 6. Advanced analyses (`06_advanced_analyses.py`)
- **Climate analysis** — monthly temperature curves by continent reveal opposite Northern/Southern-hemisphere seasonality. (`climate_seasonal_by_continent.png`)
- **Environmental impact** — PM2.5 correlates negatively with humidity (−0.29), cloud (−0.24), and precipitation (−0.21): wet, cloudy conditions coincide with cleaner air. (`airquality_correlation.png`, `airquality_by_continent.png`)
- **Feature importance** — RF importance + permutation importance for predicting temperature. **Latitude** is the dominant driver (~0.37), followed by **UV index** and **pressure** — physically sensible. (`feature_importance_rf.png`, `feature_importance_permutation.png`)
- **Spatial analysis** — global station map colored by latest temperature. (`spatial_temperature_map.png`)
- **Geographic patterns** — hottest countries are Gulf/Southeast-Asian (Qatar, UAE, Cambodia ≈ 32 °C); coldest are high-latitude. (`geographic_temperature_extremes.png`)

---

## Key insights

1. Temperature is overwhelmingly governed by **latitude and solar exposure (UV)**; pressure and humidity add secondary signal.
2. A simple **lag + seasonality** model forecasts global mean temperature to within ~0.4 °C MAE; tree ensembles cut RMSE by ~15% over linear regression.
3. **Cleaner air tracks wet weather** — precipitation and humidity wash out particulates, visible in the negative PM2.5 correlations.
4. **Data quality matters:** multilingual country duplicates and a heavily-concentrated visibility column both required targeted handling that naive pipelines would miss.

---

## Reproducibility

All randomness is seeded (`random_state=42`). Re-running `python run_all.py` regenerates `data/weather_clean.csv`, every figure, and `outputs/metrics.json` identically.

## License / attribution

Dataset © its Kaggle authors. Analysis code provided for the assessment.
