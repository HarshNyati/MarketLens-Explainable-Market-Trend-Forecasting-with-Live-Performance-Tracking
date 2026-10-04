#!/usr/bin/env python3
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.candle_ingester import run_candle_ingestion

if __name__ == "__main__":
    print("⏳ Running candle ingestion (1h candles)...")
    results = run_candle_ingestion()
    for symbol, count in results.items():
        print(f"  • {symbol}: {count} new candles inserted")
    print("✅ Candle ingestion finished successfully")
