"""Simule du trafic de production vers l'API de scoring crédit."""

import argparse
import os
import time
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def generate_client(rng: np.random.Generator, drift: bool = False) -> dict:
    """Génère des features synthétiques basées sur les distributions du dataset d'entraînement."""
    age_years = rng.uniform(25, 68) if not drift else rng.uniform(38, 74)
    days_birth = -(age_years * 365)

    employed_years = rng.uniform(0.5, min(age_years - 20, 15))
    days_employed = -(employed_years * 365)

    amt_income = max(25_650, rng.lognormal(mean=11.9, sigma=0.5))
    if drift:
        amt_income *= rng.uniform(0.65, 0.80)

    credit_months = rng.uniform(12, 72) if not drift else rng.uniform(36, 96)
    amt_credit = float(np.clip(amt_income / 12 * credit_months, 45_000, 4_050_000))

    annuity_rate = rng.uniform(0.025, 0.055) if not drift else rng.uniform(0.050, 0.080)
    amt_annuity = max(1_620, amt_credit * annuity_rate)

    ext_shift = 0.0 if not drift else -0.15
    # EXT_SOURCE_1 est absent pour ~56 % des dossiers dans les données d'entraînement
    ext1 = float(np.clip(rng.normal(0.50, 0.21) + ext_shift, 0.01, 0.99)) if rng.random() > 0.56 else None
    ext2 = float(np.clip(rng.normal(0.51, 0.19) + ext_shift, 0.01, 0.99))
    ext3 = float(np.clip(rng.normal(0.51, 0.19) + ext_shift, 0.01, 0.99)) if rng.random() > 0.20 else None

    return {
        "DAYS_BIRTH": round(days_birth, 1),
        "DAYS_EMPLOYED": round(days_employed, 1),
        "AMT_INCOME_TOTAL": round(amt_income, 2),
        "AMT_CREDIT": round(amt_credit, 2),
        "AMT_ANNUITY": round(amt_annuity, 2),
        "EXT_SOURCE_1": round(ext1, 4) if ext1 is not None else None,
        "EXT_SOURCE_2": round(ext2, 4),
        "EXT_SOURCE_3": round(ext3, 4) if ext3 is not None else None,
        "CODE_GENDER": str(rng.choice(["F", "M"], p=[0.66, 0.34])),
        "CNT_CHILDREN": int(rng.choice([0, 1, 2, 3], p=[0.70, 0.20, 0.09, 0.01])),
        "FLAG_OWN_CAR": str(rng.choice(["N", "Y"], p=[0.66, 0.34])),
        "FLAG_OWN_REALTY": str(rng.choice(["Y", "N"], p=[0.69, 0.31])),
        "AMT_GOODS_PRICE": round(float(amt_credit * rng.uniform(0.85, 1.0)), 2) if rng.random() > 0.10 else None,
        "DAYS_REGISTRATION": round(float(rng.uniform(-15_000, -300)), 1),
        "DAYS_ID_PUBLISH": round(float(rng.uniform(-5_000, -100)), 1),
        "DAYS_LAST_PHONE_CHANGE": round(float(rng.uniform(-2_500, 0)), 1),
        "REGION_RATING_CLIENT": int(rng.choice([1, 2, 3], p=[0.10, 0.74, 0.16])),
        "REGION_RATING_CLIENT_W_CITY": int(rng.choice([1, 2, 3], p=[0.10, 0.74, 0.16])),
    }


def run_simulation(n: int, drift: bool, api_url: str, api_key: str, delay: float, seed: int) -> None:
    rng = np.random.default_rng(seed)

    predict_url = f"{api_url.rstrip('/')}/predict"
    headers = {"X-API-Key": api_key}

    ok = errors = approved = rejected = 0
    latencies: list[float] = []

    print(f"{'='*60}")
    print(f"Simulation {'AVEC drift' if drift else 'sans drift'}")
    print(f"Cible : {predict_url}")
    print(f"Requêtes : {n} | seed={seed}")
    print(f"{'='*60}")

    with httpx.Client(timeout=15) as client:
        for i in range(1, n + 1):
            payload = generate_client(rng, drift=drift)
            try:
                t0 = time.perf_counter()
                resp = client.post(predict_url, json=payload, headers=headers)
                latency_ms = (time.perf_counter() - t0) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    ok += 1
                    latencies.append(latency_ms)
                    if data["decision"] == "approved":
                        approved += 1
                    else:
                        rejected += 1

                    if i % 25 == 0 or i == n:
                        print(
                            f"  [{i:4d}/{n}]  score={data['score']:.3f}"
                            f"  {data['decision']:8s}  {latency_ms:.0f}ms"
                        )
                else:
                    errors += 1
                    print(f"  [{i:4d}/{n}]  ERREUR {resp.status_code}: {resp.text[:100]}")

            except httpx.RequestError as exc:
                errors += 1
                print(f"  [{i:4d}/{n}]  EXCEPTION: {exc}")

            if delay > 0:
                time.sleep(delay)

    print(f"{'='*60}")
    print(f"Terminé  →  OK={ok}  Erreurs={errors}")
    if latencies:
        lat_sorted = sorted(latencies)
        p95_idx = int(len(lat_sorted) * 0.95)
        print(f"Taux approbation : {approved / ok:.1%}")
        print(f"Latence moyenne  : {sum(latencies)/len(latencies):.1f} ms")
        print(f"Latence p95      : {lat_sorted[p95_idx]:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simule du trafic de production vers l'API de scoring crédit",
    )
    parser.add_argument("--n", type=int, default=100, help="Nombre de requêtes (défaut: 100)")
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Active le décalage de distribution pour simuler un data drift",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="URL de base de l'API (défaut: http://localhost:8000 ou $API_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY", ""),
        help="Clé API brute pour le header X-API-Key (défaut: $API_KEY du .env)",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="Délai en secondes entre requêtes (défaut: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité (défaut: 42)")

    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key est requis (ou définissez API_KEY dans .env)")

    run_simulation(args.n, args.drift, args.api_url, args.api_key, args.delay, args.seed)


if __name__ == "__main__":
    main()
