"""
04_anomaly_detection.py
-----------------------
Advanced EDA: anomaly / outlier detection on weather observations.

Two complementary methods:
  1. Isolation Forest (multivariate) over core weather features.
  2. Z-score (univariate) on temperature as a transparent cross-check.

Outputs:
  - outputs/figures/anomaly_scatter.png  temp vs humidity, anomalies flagged
  - outputs/figures/anomaly_timeline.png anomaly count per day
  - prints the share of points flagged and a few example anomalies

Run:  python src/04_anomaly_detection.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from utils import load_clean, savefig

FEATURES = ["temperature_celsius", "humidity", "pressure_mb", "wind_kph",
            "precip_mm", "cloud", "uv_index"]


def main():
    df = load_clean().copy()
    X = StandardScaler().fit_transform(df[FEATURES])

    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    df["anomaly_iso"] = iso.fit_predict(X)  # -1 anomaly, 1 normal
    n_iso = int((df["anomaly_iso"] == -1).sum())
    print(f"Isolation Forest flagged {n_iso:,} anomalies "
          f"({n_iso / len(df) * 100:.2f}% of rows)")

    # Univariate z-score on temperature
    z = (df["temperature_celsius"] - df["temperature_celsius"].mean()) / df["temperature_celsius"].std()
    df["anomaly_z"] = (z.abs() > 3).astype(int)
    print(f"Z-score (|z|>3) flagged {int(df.anomaly_z.sum()):,} temperature anomalies")

    # Scatter: temp vs humidity with anomalies highlighted
    fig, ax = plt.subplots(figsize=(9, 6))
    normal = df[df.anomaly_iso == 1]
    anom = df[df.anomaly_iso == -1]
    ax.scatter(normal["temperature_celsius"], normal["humidity"], s=4,
               alpha=0.25, color="#3498db", label="Normal")
    ax.scatter(anom["temperature_celsius"], anom["humidity"], s=10,
               alpha=0.7, color="#e74c3c", label="Anomaly")
    ax.set_xlabel("Temperature (\u00b0C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_title("Isolation Forest anomalies: temperature vs humidity")
    ax.legend()
    savefig(fig, "anomaly_scatter.png")
    plt.close(fig)

    # Timeline of anomalies per day
    daily = (anom.groupby(anom["last_updated"].dt.date).size())
    daily.index = pd.to_datetime(daily.index)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(daily.index, daily.values, color="#e74c3c", width=1.0)
    ax.set_title("Anomalies detected per day (Isolation Forest)")
    ax.set_ylabel("Count")
    savefig(fig, "anomaly_timeline.png")
    plt.close(fig)

    print("\nExample anomalies (hottest flagged):")
    cols = ["country", "location_name", "last_updated", "temperature_celsius",
            "humidity", "wind_kph"]
    print(anom.sort_values("temperature_celsius", ascending=False)[cols]
          .head(5).to_string(index=False))


if __name__ == "__main__":
    main()
