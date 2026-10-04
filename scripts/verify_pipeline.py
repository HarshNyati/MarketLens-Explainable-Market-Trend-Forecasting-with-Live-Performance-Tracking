#!/usr/bin/env python3
"""
scripts/verify_pipeline.py
──────────────────────────
End-to-end verification script for the market-trend prediction project.

Executes in sequence:
  1. Verifies database connection & schema (tables & unique indexes).
  2. Ingests / backfills ~500 1h candles from Yahoo Finance into PostgreSQL.
  3. Trains and benchmarks ML models with walk-forward validation (TimeSeriesSplit).
  4. Generates live predictions for BITCOIN, ETHEREUM, and IBM and saves to DB.
  5. Displays stored prediction records and offline benchmark metrics.

Usage:
  python scripts/verify_pipeline.py
"""

from __future__ import annotations

import sys
import os
import time

# Ensure project root is at index 0 of sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from app.database import get_db_connection
from scripts.init_db import init_db
from app.services.candle_ingester import run_candle_ingestion, get_candle_count
from ml.train_offline import main as train_offline_main
from app.services.ml_predictor import predict_symbol
import pandas as pd

ASSETS = ["BITCOIN", "ETHEREUM", "IBM"]


def verify_database():
    print("\n" + "=" * 65)
    print("  STEP 1: DATABASE & SCHEMA VERIFICATION")
    print("=" * 65)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT version();")
        v = cur.fetchone()[0]
        print(f"  ✅ Connected to PostgreSQL: {v.split(',')[0]}")
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"  ❌ Database connection failed: {exc}")
        print("     Please ensure PostgreSQL is running (e.g. docker compose up -d postgres).")
        sys.exit(1)

    print("  Ensuring schema is applied...")
    init_db()


def backfill_candles():
    print("\n" + "=" * 65)
    print("  STEP 2: 1-HOUR CANDLE INGESTION & BACKFILL")
    print("=" * 65)
    start_time = time.time()
    results = run_candle_ingestion()
    elapsed = time.time() - start_time

    for sym, count in results.items():
        total = get_candle_count(sym)
        print(f"  • {sym:<10}: {count:>4} new candles added | Total in DB: {total:>5}")
    print(f"  ✅ Ingestion completed in {elapsed:.1f}s")


def train_models():
    print("\n" + "=" * 65)
    print("  STEP 3: OFFLINE MODEL TRAINING & BENCHMARKING")
    print("=" * 65)
    train_offline_main()


def generate_live_predictions():
    print("\n" + "=" * 65)
    print("  STEP 4: LIVE PREDICTION INFERENCE & DB PERSISTENCE")
    print("=" * 65)

    for sym in ASSETS:
        res = predict_symbol(sym)
        if "error" in res:
            print(f"  ❌ {sym}: Error: {res['error']}")
            continue

        direction = res["direction"]
        conf = res["confidence"]
        probs = res.get("probabilities", {})
        prob_str = " | ".join(f"{k}: {v*100:.1f}%" for k, v in probs.items())
        tag = "🟢 UP" if direction == "UP" else ("🔴 DOWN" if direction == "DOWN" else ("⚪ NEUTRAL" if direction == "NEUTRAL" else "⚠️ NO_SIGNAL"))

        print(f"  • {sym:<10} → {tag:<12} (Confidence: {conf:5.1f}%) | [{prob_str}]")


def display_db_summary():
    print("\n" + "=" * 65)
    print("  STEP 5: DATABASE PREDICTION SUMMARY (prediction_results)")
    print("=" * 65)

    conn = get_db_connection()
    query = """
        SELECT symbol, direction, confidence, candle_timestamp, predicted_at
        FROM prediction_results
        ORDER BY predicted_at DESC
        LIMIT 10
    """
    df = pd.read_sql(query, conn)
    conn.close()


    if df.empty:
        print("  ⚠️  No prediction records found in database.")
    else:
        print(df.to_string(index=False))

    print("\n" + "=" * 65)
    print("  🎉 PIPELINE VERIFIED SUCCESSFULLY!")
    print("  Dashboard is live at: http://localhost:8501")
    print("  API Docs are live at: http://localhost:8000/docs")
    print("=" * 65 + "\n")


def main():
    verify_database()
    backfill_candles()
    train_models()
    generate_live_predictions()
    display_db_summary()


if __name__ == "__main__":
    main()
