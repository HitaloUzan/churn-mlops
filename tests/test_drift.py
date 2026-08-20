import numpy as np
import pandas as pd
import pytest

from monitoring.drift import compute_drift

# Mirrors the real 3-critical-of-19-total-columns shape from the Telco dataset
# (see docs/decisions.md ADR-005) — enough stable filler columns that a
# blanket "% of columns" threshold and a "did a critical column drift"
# threshold can actually disagree, which is the whole point of this design.
STABLE_COLUMNS = [f"stable_{i}" for i in range(16)]


def _make_df(n, tenure_loc, monthly_loc, contract_mtm_share, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "tenure": rng.normal(tenure_loc, 5, n).clip(0, 72).round().astype(int),
            "MonthlyCharges": rng.normal(monthly_loc, 10, n).clip(0, None).round(2),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"],
                size=n,
                p=[contract_mtm_share, (1 - contract_mtm_share) / 2, (1 - contract_mtm_share) / 2],
            ),
            "Churn": rng.integers(0, 2, n),
        }
    )
    # stable filler columns: same generating process on both sides, every time
    stable_rng = np.random.default_rng(1000)
    for col in STABLE_COLUMNS:
        df[col] = stable_rng.normal(50, 5, n).round(2)
    return df


@pytest.fixture
def reference_df():
    return _make_df(n=300, tenure_loc=30, monthly_loc=65, contract_mtm_share=0.3, seed=1)


def test_compute_drift_detects_no_drift_on_identical_distribution(reference_df):
    # Same seed as reference_df (seed=1) — bit-for-bit identical data, so this
    # asserts the true-negative case with zero sampling noise. A *different*
    # seed from the same distribution can still trip a single column by chance
    # (that's a property of statistical testing, not a bug in the monitor —
    # see the critical-feature test below for the "real" positive case).
    identical = _make_df(n=300, tenure_loc=30, monthly_loc=65, contract_mtm_share=0.3, seed=1)

    result = compute_drift(reference_df, identical)

    assert result["drift_detected"] is False
    assert result["trigger_reason"] == "none"
    assert result["drift_share"] == 0.0


def test_compute_drift_flags_critical_feature_below_blanket_threshold(reference_df):
    """Only the 3 critical-ish columns shift — 3/19 ≈ 15.8%, under the 30%
    blanket threshold — so detection must come from the critical-feature path,
    not the blanket share. This is the exact scenario documented in ADR-005."""
    shifted = _make_df(n=300, tenure_loc=5, monthly_loc=110, contract_mtm_share=0.95, seed=3)

    result = compute_drift(reference_df, shifted)

    assert result["drift_detected"] is True
    assert result["trigger_reason"] == "critical_feature"
    assert result["drift_share"] < result["drift_share_threshold"]
    assert "tenure" in result["drifted_critical_features"]
    assert "MonthlyCharges" in result["drifted_critical_features"]


def test_compute_drift_drops_target_column(reference_df):
    same_dist = _make_df(n=300, tenure_loc=30, monthly_loc=65, contract_mtm_share=0.3, seed=2)

    result = compute_drift(reference_df, same_dist)

    assert "Churn" not in result["per_column_drift"]
