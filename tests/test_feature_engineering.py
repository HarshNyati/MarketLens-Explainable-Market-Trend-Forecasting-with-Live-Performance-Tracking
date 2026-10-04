import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import FEATURE_COLS, create_features


def test_feature_engineering_no_future_leakage():
    """Verify that features at time t depend strictly on past and current data (<= t).

    Altering prices or volume at future times (> t) must not change any feature value
    at time t or any prior time.
    """
    np.random.seed(42)
    n = 100
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1h")

    # Generate synthetic price and volume series
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    volumes = np.random.uniform(1000, 5000, size=n)

    df1 = pd.DataFrame({
        "recorded_at": timestamps,
        "Close": prices.copy(),
        "Volume": volumes.copy(),
    })

    # Inference mode (preserves rows without requiring forward targets)
    feat1, _ = create_features(df1, for_inference=True)

    # Pick a point in time t = 60
    t = 60
    target_ts = timestamps[t]

    # Modify all future prices and volumes strictly AFTER t (t+1 .. n)
    df2 = df1.copy(deep=True)
    df2.loc[t + 1 :, "Close"] = df2.loc[t + 1 :, "Close"] * 2.5 + 50.0
    df2.loc[t + 1 :, "Volume"] = df2.loc[t + 1 :, "Volume"] * 10.0 + 1e6

    feat2, _ = create_features(df2, for_inference=True)

    # Extract all rows up to and including time t
    sub1 = feat1[feat1["recorded_at"] <= target_ts].set_index("recorded_at")[FEATURE_COLS]
    sub2 = feat2[feat2["recorded_at"] <= target_ts].set_index("recorded_at")[FEATURE_COLS]

    assert len(sub1) > 0, "No feature rows found before target timestamp"
    assert len(sub1) == len(sub2), "Row counts mismatch between runs"

    # Verify that for every feature and every timestamp <= target_ts, values are identical
    for col in FEATURE_COLS:
        diff = (sub1[col] - sub2[col]).abs().max()
        assert np.isclose(diff, 0.0, atol=1e-7), (
            f"Feature '{col}' changed for timestamps <= {target_ts} when future data was altered! "
            f"Max absolute diff: {diff}"
        )
