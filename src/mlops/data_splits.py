"""One fixed experiment design used by every script in this project, so
`train.py` (initial deploy) and `pipeline.py` (retraining) are always
comparing apples to apples.

  full dataset
  ├── eval_holdout (20%, fixed, NEVER trained on)  <- the one constant yardstick
  └── pool (80%)
      ├── reference (70% of pool)   <- what the "already in production" model trained on
      └── current_raw (30% of pool) <- simulates data that arrives after deploy;
                                        monitoring/simulate_drift.py optionally
                                        perturbs this into `current_drifted`
"""

import pandas as pd

from mlops.config import CURRENT_FRAC_OF_POOL, EVAL_HOLDOUT_FRAC, RANDOM_STATE
from pipeline.clean import TARGET_COLUMN, clean
from pipeline.ingest import load_raw_data


def make_splits(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if df is None:
        df = clean(load_raw_data())

    eval_holdout = df.sample(frac=EVAL_HOLDOUT_FRAC, random_state=RANDOM_STATE)
    pool = df.drop(eval_holdout.index)

    current_raw = pool.sample(frac=CURRENT_FRAC_OF_POOL, random_state=RANDOM_STATE)
    reference = pool.drop(current_raw.index)

    return {
        "eval_holdout": eval_holdout.reset_index(drop=True),
        "reference": reference.reset_index(drop=True),
        "current_raw": current_raw.reset_index(drop=True),
    }


def X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
