"""
03_basic_forecast.py
--------------------
Basic time-series forecasting of global mean daily temperature using the
`last_updated` feature, as required by the assessment.

Approach (no heavy TS dependencies, fully reproducible with scikit-learn):
  - Build a daily series of global mean temperature.
  - Engineer lag features (t-1, t-2, t-3, t-7) + calendar seasonality
    (day-of-year sin/cos).
  - Train a Linear Regression model on a chronological train split.
  - Compare against a Seasonal-Naive baseline (value 7 days ago).
  - Evaluate with MAE, RMSE, and MAPE on the held-out test period.

Outputs:
  - outputs/figures/basic_forecast.png   actual vs predicted on test set
  - metrics returned to the ensemble step via outputs/metrics.json (merged)

Run:  python src/03_basic_forecast.py

Note: If you prefer a classical ARIMA/SARIMA model, install `statsmodels`
and swap in statsmodels.tsa — the data prep here is model-agnostic.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import load_clean, savefig, OUT_DIR, ensure_dirs


def build_daily_series(df):
    s = (df.groupby(df["last_updated"].dt.date)["temperature_celsius"]
           .mean().sort_index())
    s.index = pd.to_datetime(s.index)
    return s.asfreq("D").interpolate()  # regular daily grid


def make_features(s):
    d = pd.DataFrame({"y": s})
    for lag in (1, 2, 3, 7):
        d[f"lag{lag}"] = d["y"].shift(lag)
    doy = d.index.dayofyear
    d["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return d.dropna()


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_pct": round(mape, 4)}


def main():
    ensure_dirs()
    df = load_clean()
    s = build_daily_series(df)
    print(f"Daily series: {len(s)} days "
          f"({s.index.min().date()} -> {s.index.max().date()})")

    feat = make_features(s)
    X = feat.drop(columns="y")
    y = feat["y"]

    # Chronological 80/20 split (no shuffling for time series)
    split = int(len(feat) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    model = LinearRegression().fit(Xtr, ytr)
    pred = model.predict(Xte)

    # Seasonal-naive baseline: prediction = value 7 days earlier
    naive = Xte["lag7"].values

    m_model = metrics(yte.values, pred)
    m_naive = metrics(yte.values, naive)
    print("\nLinear Regression :", m_model)
    print("Seasonal-Naive    :", m_naive)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(yte.index, yte.values, label="Actual", color="#2c3e50", lw=1.4)
    ax.plot(yte.index, pred, label="Linear Regression", color="#e74c3c", lw=1.2)
    ax.plot(yte.index, naive, label="Seasonal-Naive", color="#95a5a6",
            lw=1.0, ls="--")
    ax.set_title("Basic forecast: global mean daily temperature (test set)")
    ax.set_ylabel("Temperature (\u00b0C)")
    ax.legend()
    savefig(fig, "basic_forecast.png")
    plt.close(fig)

    # Persist metrics for the comparison table / README
    out = {"basic": {"LinearRegression": m_model, "SeasonalNaive": m_naive}}
    path = OUT_DIR / "metrics.json"
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update(out)
    path.write_text(json.dumps(existing, indent=2))
    print(f"\nMetrics written -> {path.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
