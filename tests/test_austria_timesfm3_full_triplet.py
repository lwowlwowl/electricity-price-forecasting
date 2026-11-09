from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.evaluate_austria_timesfm3_mainu11_full_triplet import (  # noqa: E402
    context_and_covariates,
    native_rt_windows,
)


def test_context_stops_before_issue_and_future_starts_at_issue() -> None:
    index = pd.date_range("2025-01-01", periods=32, freq="15min", tz="UTC")
    rt = pd.Series(np.arange(len(index), dtype=float), index=index)
    covariates = pd.DataFrame(
        {
            "first": np.arange(len(index), dtype=float) + 100.0,
            "second": np.arange(len(index), dtype=float) + 200.0,
        },
        index=index,
    )
    issue = index[20]
    future_index = pd.date_range(issue, periods=4, freq="15min")

    task = context_and_covariates(
        rt, covariates, issue, future_index, context_length=8
    )

    np.testing.assert_array_equal(task["context"], np.arange(12, 20, dtype=float))
    np.testing.assert_array_equal(
        task["hist_cov"][:, 0], np.arange(112, 120, dtype=float)
    )
    np.testing.assert_array_equal(
        task["future_cov"][:, 0], np.arange(120, 124, dtype=float)
    )


def test_native_windows_use_current_hour_issue_and_quarter_offset() -> None:
    issues = pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC")
    predictions = np.stack(
        [np.arange(16, dtype=float), 100.0 + np.arange(16, dtype=float)]
    )
    quarter_index = pd.date_range(issues[0], periods=8, freq="15min")

    windows = native_rt_windows(issues, predictions, quarter_index)

    np.testing.assert_array_equal(windows[0], np.arange(16, dtype=float))
    np.testing.assert_array_equal(windows[3], np.arange(3, 16, dtype=float))
    np.testing.assert_array_equal(windows[4], 100.0 + np.arange(16, dtype=float))
    np.testing.assert_array_equal(windows[7], 100.0 + np.arange(3, 16, dtype=float))
