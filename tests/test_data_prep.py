import os
import sys
from pathlib import Path

import pandas as pd


def test_data_prep_creates_csvs(tmp_path, monkeypatch):
    # Run in isolated temp directory
    monkeypatch.chdir(tmp_path)

    # Ensure project root is on sys.path, then import to generate CSVs
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    import importlib
    data_prep = importlib.import_module("data_prep")

    # Files should exist
    assert os.path.exists(tmp_path / "reference.csv")
    assert os.path.exists(tmp_path / "current.csv")

    # Validate basic schema
    ref = pd.read_csv(tmp_path / "reference.csv")
    curr = pd.read_csv(tmp_path / "current.csv")

    expected_cols = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]
    assert list(ref.columns) == expected_cols
    assert list(curr.columns) == expected_cols

    # Non-empty and numeric features
    assert len(ref) > 0
    assert len(curr) > 0
    feature_cols = expected_cols[:-1]
    assert ref[feature_cols].dtypes.apply(lambda dt: dt.kind in "fi").all()
    assert curr[feature_cols].dtypes.apply(lambda dt: dt.kind in "fi").all()
    assert curr[feature_cols].dtypes.apply(lambda dt: dt.kind in "fi").all()
