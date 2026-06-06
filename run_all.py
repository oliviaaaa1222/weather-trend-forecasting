"""
run_all.py
----------
Runs the entire pipeline in order. From the project root:

    python run_all.py

Each stage is also runnable on its own from src/.
"""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

STAGES = [
    "01_data_cleaning.py",
    "02_eda.py",
    "03_basic_forecast.py",
    "04_anomaly_detection.py",
    "05_ensemble_forecast.py",
    "06_advanced_analyses.py",
]

if __name__ == "__main__":
    for stage in STAGES:
        print(f"\n{'='*60}\nRunning {stage}\n{'='*60}")
        runpy.run_path(str(SRC / stage), run_name="__main__")
    print("\nPipeline complete. See outputs/figures/ and outputs/metrics.json")
