"""
02_eda.py
---------
Exploratory Data Analysis on the cleaned dataset.

Produces (in outputs/figures/):
  - correlation_heatmap.png     correlations among core weather variables
  - temperature_distribution.png global temperature histogram + KDE
  - global_temp_trend.png       daily mean temperature over time
  - global_precip_trend.png     daily mean precipitation over time
  - temp_by_continent.png       temperature distribution by continent (boxplot)

Also prints a short text summary of key statistics.

Run:  python src/02_eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_clean, savefig

sns.set_theme(style="whitegrid")

CORE = [
    "temperature_celsius", "feels_like_celsius", "humidity", "precip_mm",
    "wind_kph", "pressure_mb", "cloud", "uv_index", "visibility_km",
    "air_quality_PM2.5",
]


def main():
    df = load_clean()
    print(f"Loaded clean data: {df.shape[0]:,} rows")

    # --- Summary statistics ---
    print("\nKey statistics (temperature_celsius):")
    print(df["temperature_celsius"].describe().round(2).to_string())

    # 1. Correlation heatmap
    corr = df[CORE].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap of core weather variables")
    savefig(fig, "correlation_heatmap.png")
    plt.close(fig)

    # Report strongest correlations with temperature
    tcorr = corr["temperature_celsius"].drop("temperature_celsius").sort_values(key=abs, ascending=False)
    print("\nStrongest correlations with temperature:")
    print(tcorr.round(3).to_string())

    # 2. Temperature distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["temperature_celsius"], bins=60, kde=True, color="#d1495b", ax=ax)
    ax.set_title("Global temperature distribution (\u00b0C)")
    ax.set_xlabel("Temperature (\u00b0C)")
    savefig(fig, "temperature_distribution.png")
    plt.close(fig)

    # 3 & 4. Global daily trends
    daily = df.groupby(df["last_updated"].dt.date).agg(
        temp=("temperature_celsius", "mean"),
        precip=("precip_mm", "mean"),
    )
    daily.index = __import__("pandas").to_datetime(daily.index)
    daily = daily.sort_index()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily.index, daily["temp"], color="#e67e22", lw=1.1)
    ax.set_title("Global mean daily temperature over time")
    ax.set_ylabel("Temperature (\u00b0C)")
    savefig(fig, "global_temp_trend.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily.index, daily["precip"], color="#2980b9", lw=1.1)
    ax.set_title("Global mean daily precipitation over time")
    ax.set_ylabel("Precipitation (mm)")
    savefig(fig, "global_precip_trend.png")
    plt.close(fig)

    # 5. Temperature by continent
    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = df.groupby("continent")["temperature_celsius"].median().sort_values().index
    sns.boxplot(data=df, x="continent", y="temperature_celsius", order=order,
                palette="viridis", ax=ax)
    ax.set_title("Temperature distribution by continent")
    ax.set_xlabel("")
    ax.set_ylabel("Temperature (\u00b0C)")
    plt.xticks(rotation=20)
    savefig(fig, "temp_by_continent.png")
    plt.close(fig)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
