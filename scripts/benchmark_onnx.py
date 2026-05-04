"""
Test de faisabilité ONNX Runtime — tente de convertir le pipeline joblib en ONNX
et compare les temps d'inférence si la conversion réussit.

Usage:
    python scripts/benchmark_onnx.py

Dépendances supplémentaires:
    pip install skl2onnx onnxruntime
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import feature_engineering

N_ITER = 200
MODEL_PATH = "model/model.pkl"
DATA_PATH = "data/train_sample.csv"
OUTPUT_PATH = "monitoring/benchmark_onnx.json"

_DERIVED_COLS = {
    "AGE_YEARS", "DAYS_EMPLOYED_PERC", "RATIO_ANNUITE_REVENU",
    "PAYMENT_RATE", "RATIO_CREDIT_REVENU",
}


def load_sample():
    df = pd.read_csv(DATA_PATH, nrows=1)
    drop_cols = {"SK_ID_CURR", "TARGET"} | _DERIVED_COLS
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df.iloc[0].to_dict()


def benchmark_inference(predict_fn, sample_df, n_iter=N_ITER):
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        predict_fn(sample_df)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "mean":   round(float(arr.mean()), 3),
        "median": round(float(np.median(arr)), 3),
        "p95":    round(float(np.percentile(arr, 95)), 3),
        "p99":    round(float(np.percentile(arr, 99)), 3),
    }


def main():
    print("=" * 55)
    print("  Benchmark ONNX Runtime — test de faisabilité")
    print("=" * 55)

    # --- Chargement des données ---
    sample = load_sample()
    df = pd.DataFrame([sample])
    df = feature_engineering(df)

    # --- Chargement modèle joblib (référence) ---
    print("\n[1/4] Chargement du modèle joblib...")
    t0 = time.perf_counter()
    pipeline = joblib.load(MODEL_PATH)
    joblib_load_ms = (time.perf_counter() - t0) * 1000
    print(f"      OK — {joblib_load_ms:.1f} ms")

    score_joblib = float(pipeline.predict_proba(df)[0][1])
    print(f"      Score de référence : {score_joblib:.4f}")

    # --- Conversion ONNX ---
    print("\n[2/4] Tentative de conversion ONNX...")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
        from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs

        # On infère les types depuis le dataframe après feature_engineering
        initial_types = []
        for col in df.columns:
            if df[col].dtype == object:
                initial_types.append((col, StringTensorType([None, 1])))
            else:
                initial_types.append((col, FloatTensorType([None, 1])))

        onnx_model = convert_sklearn(pipeline, initial_types=initial_types)
        print("      Conversion réussie ✓")
        conversion_ok = True
        conversion_error = None

    except Exception as e:
        print(f"      Conversion échouée ✗ — {type(e).__name__}: {e}")
        conversion_ok = False
        conversion_error = str(e)

    if not conversion_ok:
        result = {
            "run_at": datetime.now().isoformat(),
            "conversion_ok": False,
            "conversion_error": conversion_error,
        }
        with open(OUTPUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Résultats sauvegardés dans {OUTPUT_PATH}")
        return

    # --- Vérification parité des prédictions ---
    print("\n[3/4] Vérification des prédictions...")
    import onnxruntime as rt

    t0 = time.perf_counter()
    sess = rt.InferenceSession(onnx_model.SerializeToString())
    onnx_load_ms = (time.perf_counter() - t0) * 1000
    print(f"      Chargement session ONNX : {onnx_load_ms:.1f} ms")

    # Prépare l'input ONNX (format attendu par la session)
    input_feed = {
        col: df[[col]].values.astype(np.float32)
        if df[col].dtype != object
        else df[[col]].values
        for col in df.columns
    }

    onnx_output = sess.run(None, input_feed)
    # predict_proba retourne [probas_classe_0, probas_classe_1] en ONNX
    score_onnx = float(onnx_output[1][0][1]) if len(onnx_output) > 1 else float(onnx_output[0][0][1])

    diff = abs(score_joblib - score_onnx)
    parity_ok = diff < 1e-4
    print(f"      joblib : {score_joblib:.6f}")
    print(f"      onnx   : {score_onnx:.6f}")
    print(f"      écart  : {diff:.2e} — {'OK ✓' if parity_ok else 'ATTENTION ✗'}")

    # --- Benchmark ---
    print(f"\n[4/4] Benchmark {N_ITER} itérations...")

    def joblib_predict(d):
        pipeline.predict_proba(d)

    def onnx_predict(d):
        sess.run(None, input_feed)

    stats_joblib = benchmark_inference(joblib_predict, df)
    stats_onnx   = benchmark_inference(onnx_predict, df)

    gain_mean = (stats_onnx["mean"] - stats_joblib["mean"]) / stats_joblib["mean"] * 100
    gain_p95  = (stats_onnx["p95"]  - stats_joblib["p95"])  / stats_joblib["p95"]  * 100

    print(f"\n  {'Métrique':<20} {'joblib':>8} {'onnx':>8} {'gain':>8}")
    print(f"  {'-'*48}")
    print(f"  {'Inférence mean (ms)':<20} {stats_joblib['mean']:>8.3f} {stats_onnx['mean']:>8.3f} {gain_mean:>+7.1f}%")
    print(f"  {'Inférence p95  (ms)':<20} {stats_joblib['p95']:>8.3f} {stats_onnx['p95']:>8.3f} {gain_p95:>+7.1f}%")

    result = {
        "run_at": datetime.now().isoformat(),
        "n_iter": N_ITER,
        "conversion_ok": True,
        "parity_ok": parity_ok,
        "score_joblib": round(score_joblib, 6),
        "score_onnx":   round(score_onnx, 6),
        "load_time_ms": {
            "joblib": round(joblib_load_ms, 1),
            "onnx":   round(onnx_load_ms, 1),
        },
        "inference_joblib": stats_joblib,
        "inference_onnx":   stats_onnx,
        "gain_mean_pct": round(gain_mean, 2),
        "gain_p95_pct":  round(gain_p95, 2),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Résultats sauvegardés dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
