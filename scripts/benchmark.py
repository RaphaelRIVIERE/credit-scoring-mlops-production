"""
Benchmark mlflow.pyfunc vs joblib direct.

Usage:
    python scripts/benchmark.py
    python -m cProfile -s cumtime scripts/benchmark.py 2>&1 | head -30
"""

import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import psutil
import mlflow.pyfunc

from src.preprocessing import feature_engineering

N_ITER = 200
MODEL_PATH = "model"

# Colonnes dérivées calculées par feature_engineering (à exclure du sample brut)
_DERIVED_COLS = {
    "AGE_YEARS", "DAYS_EMPLOYED_PERC", "RATIO_ANNUITE_REVENU",
    "PAYMENT_RATE", "RATIO_CREDIT_REVENU",
}

def _load_sample() -> dict:
    """Charge une ligne réelle depuis train_sample.csv (hors TARGET/ID/dérivées)."""
    df = pd.read_csv("data/train_sample.csv", nrows=1)
    drop_cols = {"SK_ID_CURR", "TARGET"} | _DERIVED_COLS
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df.iloc[0].to_dict()


def _stats(values: list[float], label: str, unit: str = "ms") -> dict:
    arr = np.array(values)
    result = {
        "mean": arr.mean(),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
    print(f"\n  {label} ({unit})")
    print(f"    mean : {result['mean']:.3f}")
    print(f"    median : {result['median']:.3f}")
    print(f"    p95 : {result['p95']:.3f}")
    print(f"    p99 : {result['p99']:.3f}")
    return result


def run_benchmark(model, sample: dict) -> dict:
    preprocess_times: list[float] = []
    inference_times: list[float] = []
    cpu_usages: list[float] = []

    for _ in range(N_ITER):
        # Preprocessing
        t0 = time.perf_counter()
        df = pd.DataFrame([sample])
        df = feature_engineering(df)
        preprocess_times.append((time.perf_counter() - t0) * 1000)

        # Inférence + CPU
        psutil.cpu_percent(interval=None)  # reset le compteur CPU
        t0 = time.perf_counter()
        model.predict(df)
        inference_times.append((time.perf_counter() - t0) * 1000)
        cpu_usages.append(psutil.cpu_percent(interval=None))

    return {
        "preprocess": preprocess_times,
        "inference": inference_times,
        "cpu": cpu_usages,
    }


def main():
    sample = _load_sample()

    print("=" * 55)
    print(f"  Benchmark mlflow.pyfunc — {N_ITER} itérations")
    print("=" * 55)

    # Chargement du modèle (mesuré une seule fois)
    t0 = time.perf_counter()
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    load_time_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  Chargement du modèle : {load_time_ms:.1f} ms")

    results = run_benchmark(model, sample)
    stats_preprocess = _stats(results["preprocess"], "Préprocessing")
    stats_inference = _stats(results["inference"], "Inférence")
    stats_cpu = _stats(results["cpu"], "CPU pendant l'inférence", unit="%")

    print("\n" + "=" * 55)
    print("  Score de référence (1ère prédiction) :")
    df = pd.DataFrame([sample])
    df = feature_engineering(df)
    score = float(model.predict(df)[0][1])
    print(f"    score = {score:.4f}")
    print("=" * 55)

    output = {
        "run_at": datetime.now().isoformat(),
        "loader": "mlflow.pyfunc",
        "n_iter": N_ITER,
        "load_time_ms": round(load_time_ms, 3),
        "preprocess": stats_preprocess,
        "inference": stats_inference,
        "cpu": stats_cpu,
        "ref_score": round(score, 4),
    }
    loader_slug = output["loader"].replace(".", "_").replace("/", "_")
    out_path = f"monitoring/benchmark_{loader_slug}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Résultats sauvegardés dans {out_path}")


if __name__ == "__main__":
    main()
