"""Tests for the feature engineering pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from feature_engineering_toolkit import FeatureEngineeringPipeline

def make_sample():
    np.random.seed(42)
    return pd.DataFrame({
        "user_id": list(range(1, 21)) * 3,
        "age": list(range(20, 40)) * 3,
        "income": [50000 + i*1000 for i in range(60)],
        "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "target": [0, 1] * 30,
    })

def test_create_features():
    df = make_sample()
    p = FeatureEngineeringPipeline(df, target_col="target")
    p.create_features()
    assert "age_sq" in p.df.columns
    assert "interaction" in p.df.columns
    assert "user_id_mean" in p.df.columns
    assert "date_year" in p.df.columns
    print("PASS: test_create_features")

def test_transform_standard():
    df = make_sample()
    p = FeatureEngineeringPipeline(df, target_col="target")
    p.transform()
    assert "age" in p.df.columns
    print("PASS: test_transform_standard")

def test_select_kbest():
    df = make_sample()
    p = FeatureEngineeringPipeline(df, target_col="target")
    p.create_features().transform().select_features()
    assert len(p.selected_features) > 0
    print("PASS: test_select_kbest")

def test_run_full():
    df = make_sample()
    p = FeatureEngineeringPipeline(df, target_col="target")
    p.run()
    assert p.df is not None
    assert p.accuracy is not None
    print("PASS: test_run_full")

if __name__ == "__main__":
    test_create_features()
    test_transform_standard()
    test_select_kbest()
    test_run_full()
    print("All feature engineering tests passed!")
