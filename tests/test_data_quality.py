import numpy as np
import pandas as pd

from autopilot.data.providers import repair_splits


def series(prices):
    idx = pd.bdate_range("2019-01-01", periods=len(prices))
    p = np.array(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 1000.0}, index=idx)


def test_unadjusted_1_to_10_split_is_repaired():
    raw = series([100 + i * 0.1 for i in range(30)] + [10.3 + i * 0.01 for i in range(30)])
    fixed, fixes = repair_splits(raw, "NIFTYBEES")
    assert fixes and fixes[0]["factor"] == 10
    worst_day = fixed["close"].pct_change().min()
    assert worst_day > -0.05                       # the fake -90% crash is gone
    assert fixed["volume"].iloc[0] == 10_000       # volume scaled the other way


def test_real_crash_that_recovers_is_left_alone():
    raw = series([100.0] * 20 + [55.0] + [95.0] * 20)   # one-day spike down, recovers
    fixed, fixes = repair_splits(raw, "X")
    assert not fixes and fixed["close"].iloc[20] == 55.0


def test_odd_ratio_crash_is_not_treated_as_split():
    raw = series([100.0] * 20 + [37.0] * 20)             # -63%: no clean split ratio nearby
    _, fixes = repair_splits(raw, "X")
    assert not fixes
