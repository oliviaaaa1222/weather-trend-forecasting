"""
06_advanced_analyses.py
-----------------------
Unique / advanced analyses required by the assessment:

  A. Climate analysis    - seasonal (monthly) temperature patterns by continent.
  B. Environmental impact- air quality (PM2.5) vs weather parameters: a
                           correlation bar chart + PM2.5 by continent.
  C. Feature importance  - Random Forest importances + permutation importance
                           for predicting temperature.
  D. Spatial analysis    - global scatter map of stations colored by temperature.
  E. Geographic patterns - mean temperature & PM2.5 ranked by country/continent.

Outputs go to outputs/figures/. A text digest is printed for the README.

Run:  python src/06_advanced_analyses.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from utils import load_clean, savefig

sns.set_theme(style="whitegrid")


def climate_analysis(df):
    piv = (df.groupby(["continent", "month"])["temperature_celsius"]
             .mean().unstack(0))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col in piv.columns:
        ax.plot(piv.index, piv[col], marker="o", ms=3, label=col)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean temperature (\u00b0C)")
    ax.set_title("Seasonal temperature pattern by continent")
    ax.legend(fontsize=8)
    savefig(fig, "climate_seasonal_by_continent.png")
    plt.close(fig)


def air_quality_analysis(df):
    aq_targets = ["temperature_celsius", "humidity", "wind_kph", "precip_mm",
                  "pressure_mb", "cloud", "visibility_km"]
    corrs = df[aq_targets + ["air_quality_PM2.5"]].corr()["air_quality_PM2.5"].drop("air_quality_PM2.5")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    corrs.sort_values().plot(kind="barh", color="#16a085", ax=ax)
    ax.set_title("Correlation of weather parameters with PM2.5")
    ax.set_xlabel("Pearson r")
    savefig(fig, "airquality_correlation.png")
    plt.close(fig)

    cont = df.groupby("continent")["air_quality_PM2.5"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cont.plot(kind="bar", color="#c0392b", ax=ax)
    ax.set_title("Mean PM2.5 by continent")
    ax.set_ylabel("PM2.5")
    plt.xticks(rotation=20)
    savefig(fig, "airquality_by_continent.png")
    plt.close(fig)
    print("PM2.5 correlations with weather:\n", corrs.round(3).to_string())


def feature_importance(df):
    feats = ["humidity", "pressure_mb", "wind_kph", "precip_mm", "cloud",
             "uv_index", "visibility_km", "latitude", "longitude", "month",
             "air_quality_PM2.5"]
    X, y = df[feats], df["temperature_celsius"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)

    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", color="#8e44ad", ax=ax)
    ax.set_title("Random Forest feature importance (predicting temperature)")
    ax.set_xlabel("Importance")
    savefig(fig, "feature_importance_rf.png")
    plt.close(fig)

    perm = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=42, n_jobs=-1)
    pimp = pd.Series(perm.importances_mean, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    pimp.plot(kind="barh", color="#2980b9", ax=ax)
    ax.set_title("Permutation importance (predicting temperature)")
    ax.set_xlabel("Mean importance")
    savefig(fig, "feature_importance_permutation.png")
    plt.close(fig)
    print("Top RF features:\n", imp.sort_values(ascending=False).head(5).round(4).to_string())


def spatial_analysis(df):
    latest = df.sort_values("last_updated").groupby("location_name").tail(1)
    fig, ax = plt.subplots(figsize=(12, 6))
    sc = ax.scatter(latest["longitude"], latest["latitude"],
                    c=latest["temperature_celsius"], cmap="RdYlBu_r",
                    s=18, alpha=0.85)
    plt.colorbar(sc, label="Temperature (\u00b0C)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial distribution of stations (latest reading, colored by temp)")
    savefig(fig, "spatial_temperature_map.png")
    plt.close(fig)


def geographic_patterns(df):
    top_hot = df.groupby("country")["temperature_celsius"].mean().sort_values(ascending=False).head(10)
    top_cold = df.groupby("country")["temperature_celsius"].mean().sort_values().head(10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    top_hot.sort_values().plot(kind="barh", color="#e74c3c", ax=axes[0])
    axes[0].set_title("Top 10 hottest countries (mean \u00b0C)")
    top_cold.sort_values(ascending=False).plot(kind="barh", color="#3498db", ax=axes[1])
    axes[1].set_title("Top 10 coldest countries (mean \u00b0C)")
    fig.tight_layout()
    savefig(fig, "geographic_temperature_extremes.png")
    plt.close(fig)
    print("Hottest countries:\n", top_hot.round(1).to_string())


def main():
    df = load_clean()
    print("== A. Climate analysis =="); climate_analysis(df)
    print("\n== B. Air quality / environmental impact =="); air_quality_analysis(df)
    print("\n== C. Feature importance =="); feature_importance(df)
    print("\n== D. Spatial analysis =="); spatial_analysis(df)
    print("\n== E. Geographic patterns =="); geographic_patterns(df)
    print("\nAdvanced analyses complete.")


if __name__ == "__main__":
    main()
