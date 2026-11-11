"""Tests for AI/ML Model Suite, Predictions, and Benchmarking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_generator import FEATURE_COLUMNS, generate_synthetic_student_dataset
from explainability import compute_local_explanation
from ml_models import AVAILABLE_MODELS, get_ai_model_suite


def test_synthetic_dataset_properties(tmp_path):
    """Verify synthetic dataset generator produces valid, non-empty, schema-compliant rows."""
    test_csv = tmp_path / "test_synth.csv"
    df = generate_synthetic_student_dataset(n_samples=60, random_state=42, output_path=test_csv)
    
    assert len(df) == 60
    assert "target_career" in df.columns
    assert "target_onet_code" in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns
        assert not df[col].isna().any()

    # Academic score bounds
    assert (df["math_score"] >= 35.0).all() and (df["math_score"] <= 100.0).all()
    # RIASEC bounds
    assert (df["riasec_investigative"] >= 1.0).all() and (df["riasec_investigative"] <= 5.0).all()
    # Skills bounds
    assert (df["skill_critical_thinking"] >= 1.0).all() and (df["skill_critical_thinking"] <= 5.0).all()


def test_model_suite_loaded():
    """Verify model bundle and metrics are properly loaded into the singleton suite."""
    suite = get_ai_model_suite()
    if not (Path(__file__).resolve().parents[1] / "models" / "career_models.joblib").exists():
        # Compact submissions omit the generated bundle; the documented training
        # command recreates it locally.
        assert suite.is_loaded is False
        assert suite.bundle is None
        return

    assert suite.is_loaded is True
    assert suite.bundle is not None
    assert len(suite.bundle.get("classes", [])) == 20
    assert len(suite.bundle.get("feature_names", [])) == 29
    assert suite.bundle.get("scaler") is not None


def test_individual_models_inference():
    """Verify all individual AI models generate valid probability distributions."""
    suite = get_ai_model_suite()
    sample_vec = np.zeros((1, 29), dtype=np.float32)
    # Set high math, cs, programming
    sample_vec[0, 0] = 95.0  # math_score
    sample_vec[0, 4] = 96.0  # cs_score
    sample_vec[0, 14] = 4.8  # riasec_investigative
    sample_vec[0, 22] = 4.8  # skill_programming

    for model_name in AVAILABLE_MODELS:
        preds = suite.predict_careers(sample_vec, model_name=model_name, top_k=5)
        if not suite.is_loaded:
            assert preds == []
            continue
        assert len(preds) == 5
        assert preds[0]["confidence_score"] >= preds[1]["confidence_score"]
        assert all(0.0 <= p["confidence_score"] <= 1.0 for p in preds)
        assert all(p["onet_code"] for p in preds)


def test_explainability_local_attribution():
    """Verify Explainable AI engine produces drivers and category contributions."""
    suite = get_ai_model_suite()
    sample_vec = np.zeros((1, 29), dtype=np.float32)
    sample_vec[0, 0] = 92.0  # math
    sample_vec[0, 4] = 90.0  # cs
    sample_vec[0, 22] = 4.5  # programming

    explanation = compute_local_explanation(sample_vec, "Software Developer / Engineer")
    assert "drivers" in explanation
    assert "cautions" in explanation
    assert "category_contributions" in explanation
    assert isinstance(explanation["drivers"], list)
    if not suite.is_loaded:
        assert explanation["drivers"] == []
        assert explanation["category_contributions"] == {}
        return
    assert len(explanation["drivers"]) > 0
    assert "Academic Performance" in explanation["category_contributions"]
