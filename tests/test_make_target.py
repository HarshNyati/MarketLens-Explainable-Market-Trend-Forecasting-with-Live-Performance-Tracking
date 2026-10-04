import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import make_target


def test_make_target_up_down_neutral():
    """Verify that make_target labels match the +/-0.5% rule over the horizon."""
    # Base price = 100.0, horizon = 5, up_thresh = 0.005 (+0.5%), dn_thresh = 0.005 (-0.5%)
    # Sequence of prices:
    # t=0: 100.0 -> next 5 prices: [100.1, 100.2, 101.0, 100.3, 100.2]
    #               max_ret = +1.0% (>= +0.5%), min_ret = +0.1% -> UP (2)
    # t=1: 100.0 -> next 5 prices: [99.8, 99.7, 99.0, 99.4, 99.5]
    #               max_ret = -0.2%, min_ret = -1.0% (<= -0.5%) -> DOWN (0)
    # t=2: 100.0 -> next 5 prices: [100.1, 99.9, 100.2, 99.8, 100.0]
    #               max_ret = +0.2%, min_ret = -0.2% (within +/-0.5%) -> NEUTRAL (1)
    # t=3: 100.0 -> next 5 prices: [102.0, 99.0, 100.0, 100.0, 100.0]
    #               max_ret = +2.0%, min_ret = -1.0% -> abs(max) > abs(min) -> UP (2)
    # t=4: 100.0 -> next 5 prices: [101.0, 97.0, 100.0, 100.0, 100.0]
    #               max_ret = +1.0%, min_ret = -3.0% -> abs(min) > abs(max) -> DOWN (0)

    prices = pd.Series([
        100.0,  # t=0
        100.0,  # t=1
        100.0,  # t=2
        100.0,  # t=3
        100.0,  # t=4
        101.0,  # t=5
        99.0,   # t=6
        100.0,  # t=7
        100.0,  # t=8
        100.0,  # t=9
    ])

    # Let's construct a cleaner explicit test for 5 rows with 5 future bars each:
    # Explicit series:
    # row 0: P0 = 100, future = [100.1, 100.2, 100.8, 100.3, 100.1] -> max +0.8%, min +0.1% -> UP (2)
    # row 1: P0 = 100, future = [99.8, 99.7, 99.1, 99.5, 99.6] -> min -0.9%, max -0.2% -> DOWN (0)
    # row 2: P0 = 100, future = [100.2, 99.8, 100.1, 99.9, 100.3] -> max +0.3%, min -0.2% -> NEUTRAL (1)
    # row 3: P0 = 100, future = [101.5, 98.9, 100.0, 100.0, 100.0] -> max +1.5%, min -1.1% -> UP (2)
    # row 4: P0 = 100, future = [100.9, 97.5, 100.0, 100.0, 100.0] -> max +0.9%, min -2.5% -> DOWN (0)

    # Let's test single-row predictions directly using sub-series to avoid overlap confusion:
    # UP case
    s_up = pd.Series([100.0, 100.1, 100.2, 100.8, 100.3, 100.1])
    target_up = make_target(s_up, horizon=5, up_thresh=0.005, dn_thresh=0.005)
    assert target_up.iloc[0] == 2  # UP

    # DOWN case
    s_down = pd.Series([100.0, 99.9, 99.8, 99.1, 99.6, 99.7])
    target_down = make_target(s_down, horizon=5, up_thresh=0.005, dn_thresh=0.005)
    assert target_down.iloc[0] == 0  # DOWN

    # NEUTRAL case (< 0.5% move)
    s_neutral = pd.Series([100.0, 100.2, 99.8, 100.1, 99.9, 100.3])
    target_neutral = make_target(s_neutral, horizon=5, up_thresh=0.005, dn_thresh=0.005)
    assert target_neutral.iloc[0] == 1  # NEUTRAL

    # Last `horizon` rows must be NaN
    assert np.isnan(target_up.iloc[1:]) .all()
