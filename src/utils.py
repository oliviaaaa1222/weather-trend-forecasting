"""Shared helpers: paths, IO, and small reusable functions."""
from pathlib import Path
import pandas as pd

# Project paths (resolved relative to this file so scripts run from anywhere)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

RAW_CSV = DATA_DIR / "GlobalWeatherRepository.csv"
CLEAN_CSV = DATA_DIR / "weather_clean.csv"

# Continent lookup keyed by country, used for the geographic analyses.
# (A pragmatic mapping covering the countries present in the dataset.)
CONTINENT = {}


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    df = pd.read_csv(RAW_CSV)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    return df


def load_clean():
    df = pd.read_csv(CLEAN_CSV)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    return df


def savefig(fig, name):
    ensure_dirs()
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  saved figure -> {path.relative_to(ROOT)}")
    return path
