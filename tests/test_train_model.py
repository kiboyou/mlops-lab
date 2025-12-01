import os
import sys
from pathlib import Path

import joblib
import pandas as pd


def test_train_model_creates_artifact(tmp_path, monkeypatch):
    # Run in isolated temp directory
    monkeypatch.chdir(tmp_path)

    # Prepare reference/current CSVs via data_prep (force reload to run top-level code)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    import importlib
    if "data_prep" in sys.modules:
        importlib.reload(sys.modules["data_prep"])
    else:
        importlib.import_module("data_prep")

    # Train model
    importlib.import_module("train_model")

    # Model artifact should exist
    artifact_path = tmp_path / "artifacts" / "model.joblib"
    assert os.path.exists(artifact_path)

    # Load model and perform a quick prediction on current data
    model = joblib.load(artifact_path)
    curr = pd.read_csv(tmp_path / "current.csv")
    X = curr.drop("target", axis=1)
    preds = model.predict(X)

    # Sanity checks
    assert len(preds) == len(curr)
    # Iris targets are 0/1/2
    assert set(preds).issubset({0, 1, 2})
    assert set(preds).issubset({0, 1, 2})
