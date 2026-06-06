"""
05_ensemble_forecast.py
-----------------------
Forecasting with multiple models + an ensemble.

Models compared on the global daily mean-temperature series:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Ensemble = simple average of the three model predictions

Same lag + seasonal features and chronological split as the basic model,
so results are directly comparable. Metrics (MAE/RMSE/MAPE) for every model
are written to outputs/metrics.json and the comparison is plotted.

Run:  python src/05_ensemble_forecast.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import load_clean, savefig, OUT_DIR, ensure_dirs


def build_features(df):
    s = (df.groupby(df["last_updated"].dt.date)["temperature_celsius"]
           .mean().sort_index())
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("D").interpolate()
    d = pd.DataFrame({"y": s})
    for lag in (1, 2, 3, 7, 14):
        d[f"lag{lag}"] = d["y"].shift(lag)
    d["roll7"] = d["y"].shift(1).rolling(7).mean()
    doy = d.index.dayofyear
    d["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return d.dropna()


def metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAPE_pct": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 4),
    }


def main():
    ensure_dirs()
    df = load_clean()
    feat = build_features(df)
    X, y = feat.drop(columns="y"), feat["y"]
    split = int(len(feat) * 0.8)
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
    }

    preds, results = {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p = mdl.predict(Xte)
        preds[name] = p
        results[name] = metrics(yte.values, p)
        print(f"{name:18s}: {results[name]}")

    # Ensemble: average of the three
    ens = np.mean(np.column_stack(list(preds.values())), axis=1)
    results["Ensemble_Mean"] = metrics(yte.values, ens)
    print(f"{'Ensemble_Mean':18s}: {results['Ensemble_Mean']}")

    best = min(results, key=lambda k: results[k]["RMSE"])
    print(f"\nBest model by RMSE: {best}")

    # Plot actual vs ensemble + best single model
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(yte.index, yte.values, label="Actual", color="#2c3e50", lw=1.5)
    ax.plot(yte.index, ens, label="Ensemble (mean)", color="#27ae60", lw=1.2)
    ax.set_title("Ensemble forecast vs actual (test set)")
    ax.set_ylabel("Temperature (\u00b0C)")
    ax.legend()
    savefig(fig, "ensemble_forecast.png")
    plt.close(fig)

    # Bar chart of RMSE across models
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(results.keys())
    rmses = [results[n]["RMSE"] for n in names]
    ax.bar(names, rmses, color=["#3498db", "#9b59b6", "#e67e22", "#27ae60"])
    ax.set_ylabel("RMSE (\u00b0C)")
    ax.set_title("Model comparison (lower is better)")
    plt.xticks(rotation=15)
    savefig(fig, "model_comparison.png")
    plt.close(fig)

    # Persist
    path = OUT_DIR / "metrics.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing["advanced_models"] = results
    existing["best_model"] = best
    path.write_text(json.dumps(existing, indent=2))
    print(f"Metrics written -> {path.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
