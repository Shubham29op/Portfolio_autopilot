import numpy as np
import pandas as pd

from autopilot.config import Universe
from autopilot.data.providers import SyntheticProvider
from autopilot.features import FEATURES, build_features
from autopilot.ml.labels import triple_barrier


def test_features_do_not_use_future_data():
    u = Universe.load()
    syms = ["NIFTYBEES", "INFY", "TCS", "GOLDBEES"]
    bars = SyntheticProvider(end="2020-12-31").load(syms)
    full = build_features(bars, "NIFTYBEES")
    cut = pd.Timestamp("2019-06-28")
    trunc = build_features({s: df[df.index <= cut] for s, df in bars.items()}, "NIFTYBEES")
    a = full.xs(cut, level="date")[FEATURES].sort_index()
    b = trunc.xs(cut, level="date")[FEATURES].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_triple_barrier_hits():
    idx = pd.bdate_range("2024-01-01", periods=8)
    df = pd.DataFrame({"open": [100] * 8, "high": [101, 101, 107, 101, 101, 101, 101, 101],
                       "low": [99] * 8, "close": [100] * 8}, index=idx)
    atr = pd.Series(2.0, index=idx)
    lab = triple_barrier(df, atr, horizon=3, stop_atr=2, target_atr=3)
    assert lab["label"].iloc[0] == 1          # target 106 hit on day 3
    assert np.isnan(lab["label"].iloc[-1])    # not enough future
