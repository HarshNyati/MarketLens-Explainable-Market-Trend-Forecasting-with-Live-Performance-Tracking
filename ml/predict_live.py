"""
ml/predict_live.py
───────────────────
Standalone script to run live predictions for all tracked assets.
Uses app.services.ml_predictor.predict_symbol() to ensure zero drift
between the CLI script and the FastAPI endpoint.
"""

from __future__ import annotations

import sys
import os

# Allow running as a standalone script from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.ml_predictor import predict_symbol

ASSETS = ["BITCOIN", "ETHEREUM", "IBM"]


def main():
    print("=" * 60)
    print("  LIVE PREDICTIONS (using ingested DB candles & calibrated ML)")
    print("=" * 60)

    for asset in ASSETS:
        print(f"\n🔮 Evaluating {asset}...")
        res = predict_symbol(asset)

        if "error" in res:
            print(f"  ❌ Error: {res['error']}")
            continue

        direction = res["direction"]
        conf = res["confidence"]
        probs = res.get("probabilities", {})
        prob_str = " | ".join(f"{k}: {v*100:.1f}%" for k, v in probs.items())

        print(f"  • Direction : {direction}")
        print(f"  • Confidence: {conf:.1f}%")
        print(f"  • Probabilities: {prob_str}")
        print(f"  • Candles analyzed: {res.get('candles_used')}")
        if not res.get("is_signal"):
            print(f"  • Note: {res.get('reason')}")

    print("\n✅ Live prediction complete. Results persisted to database.")


if __name__ == "__main__":
    main()
