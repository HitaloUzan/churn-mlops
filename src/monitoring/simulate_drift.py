"""Simulates "what if the data changes after deploy?" — the scenario this
whole project exists to detect and react to (see docs/decisions.md ADR-003).

`current_raw` (from mlops.data_splits.make_splits) is real held-out Telco data
with the same distribution as `reference` — i.e., no drift. `inject_drift`
perturbs it into `current_drifted`, a controlled, known distribution shift, so
the drift monitor's before/after behavior is verifiable, not just plausible.
"""

import numpy as np
import pandas as pd

from mlops.config import RANDOM_STATE


def inject_drift(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Simulate a plausible real-world shift: a promotional push toward
    month-to-month contracts drives in a wave of new, lower-tenure, higher
    monthly-charge customers. This is a big, deliberately obvious shift so
    the monitor's detection is unambiguous in the demo — see ADR-003 for why
    the magnitude was chosen this way.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    n = len(df)
    switch_to_mtm = rng.random(n) < 0.4
    df.loc[switch_to_mtm, "Contract"] = "Month-to-month"

    df["tenure"] = (df["tenure"] * 0.5).round().clip(lower=0).astype(int)

    charge_multiplier = rng.normal(loc=1.35, scale=0.1, size=n).clip(1.0, None)
    df["MonthlyCharges"] = (df["MonthlyCharges"] * charge_multiplier).round(2)

    return df
