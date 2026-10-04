"""
ml/visualize.py
────────────────
Visualise model performance on a tail of each historical CSV.
Uses the same feature_engineering pipeline as training/inference.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ml.feature_engineering import FEATURE_COLS, create_features

warnings.filterwarnings("ignore")

ASSETS = {
    "BITCOIN":  "ml/datasets/btc_historical.csv",
    "ETHEREUM": "ml/datasets/eth_historical.csv",
    "IBM":      "ml/datasets/ibm_historical.csv",
}


def visualize_asset(symbol: str, csv_path: str) -> None:
    print(f"\n📊 Visualising {symbol}")

    df = pd.read_csv(csv_path)
    df = df.tail(2000)

    # Normalise column names
    for src, dst in [("close", "Close"), ("Price", "Close"), ("Value", "Close")]:
        if src in df.columns and "Close" not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    df, _ = create_features(df)

    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"],
    )
    plt.title(f"{symbol} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ── Price + prediction markers ─────────────────────────────────────────────
    price_test = df["Close"].iloc[-len(y_test):].reset_index(drop=True)

    plt.figure(figsize=(12, 4))
    plt.plot(price_test.values, label="Price", color="black")

    up_idx   = preds == 1
    down_idx = preds == 0

    plt.scatter(price_test.index[up_idx],   price_test[up_idx],
                color="green", label="Predicted UP",   marker="^")
    plt.scatter(price_test.index[down_idx], price_test[down_idx],
                color="red",   label="Predicted DOWN", marker="v")

    plt.title(f"{symbol} — Price with Prediction Signals")
    plt.xlabel("Time Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for sym, path in ASSETS.items():
        visualize_asset(sym, path)
