#!/usr/bin/env python
"""Read-only XGBoost 3.4.0 compatibility worker for frozen joblib models."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / ".venv" / "Lib" / "site-packages"))

# Load the project's scikit-learn first.  XGBoost itself is still resolved from
# the isolated interpreter because its site-packages precedes the appended path.
import sklearn  # noqa: F401,E402
import xgboost  # noqa: F401,E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = joblib.load(args.checkpoint)
    with np.load(args.input, allow_pickle=False) as archive:
        features = archive["features"]
    prediction = payload["estimator"].predict(features).astype(np.float32)
    np.savez_compressed(args.output, prediction=prediction)


if __name__ == "__main__":
    main()
