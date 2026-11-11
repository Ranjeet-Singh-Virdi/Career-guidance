"""Automated tests for career_engine scoring, weight redistribution, and explainability."""
from __future__ import annotations

import pytest

from assessment import EducationalStage, ExtracurricularActivity, StudentProfile, SubjectPreference
from career_engine import (
    DEFAULT_WEIGHTS,
    load_onet_data,
    recommend,
    recommend_student_profile,
    redistribute_weights,
)
from data_loader import ONET_DIR, load_all_curated_data


@pytest.fixture(scope="module")
def onet_data():
    return load_onet_data(ONET_DIR)


@pytest.fixture(scope="module")
def curated_data():
    return load_all_curated_data()


def test_legacy_recommend_signature(onet_data):
    """Verify exact backward compatibility for existing recommend() calls."""
    interests = ["Investigative", "Artistic"]
    skills = {"Critical Thinking": 4, "Reading Comprehension": 4}
    subjects = {"Mathematics": 85, "Science": 80}

    results = recommend(onet_data, interests, skills, subjects, limit=5)

    assert isinstance(results, list)
    assert len(results) == 5
    for rec in results:
        assert "code" in rec
        assert "title" in rec
        assert "score" in rec
        assert 0.0 <= rec["score"] <= 1.0
        assert "skills" in rec
        assert "weights" in rec
        assert abs(sum(rec["weights"].values()) - 1.0) < 1e-4


def test_weight_redistribution_all_active():
    """Verify weight redistribution when all 6 dimensions are active."""
    active = {
        "interests": True,
        "academics": True,
        "skills": True,
        "extracurriculars": True,
        "work_preferences": True,
        "career_priorities": True,
    }
    w = redistribute_weights(active, DEFAULT_WEIGHTS)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    for k in DEFAULT_WEIGHTS:
        assert w[k] > 0


def test_weight_redistribution_partial_skipped():
    """Verify weight redistribution when dimensions are skipped (e.g. no extracurriculars or priorities)."""
    active = {
        "interests": True,
        "academics": True,
        "skills": True,
        "extracurriculars": False,
        "work_preferences": False,
        "career_priorities": False,
    }
    w = redistribute_weights(active, DEFAULT_WEIGHTS)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["extracurriculars"] == 0.0
    assert w["work_preferences"] == 0.0
    assert w["career_priorities"] == 0.0
    # Remaining weights should be scaled proportionally
    expected_sum = DEFAULT_WEIGHTS["interests"] + DEFAULT_WEIGHTS["academics"] + DEFAULT_WEIGHTS["skills"]
    assert abs(w["interests"] - (DEFAULT_WEIGHTS["interests"] / expected_sum)) < 1e-6


def test_weight_redistribution_empty_dimensions():
    """Verify graceful fallback when all dimensions are skipped."""
    active = {k: False for k in DEFAULT_WEIGHTS}
    w = redistribute_weights(active, DEFAULT_WEIGHTS)
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_student_profile_scoring_bounds(onet_data, curated_data):
    """Verify all composite and dimension scores are strictly bounded between 0.0 and 1.0."""
    prof = StudentProfile(
        nickname="BoundaryTester",
        stage=EducationalStage.AFTER_CLASS_12.value,
        interests={"Technology and computing": 5, "Science and research": 5},
        subjects={
            "Mathematics": SubjectPreference(name="Mathematics", interest_level=5, confidence_level=5, academic_performance=100.0),
            "Physics": SubjectPreference(name="Physics", interest_level=5, confidence_level=5, academic_performance=100.0),
        },
        skills={"Critical Thinking": 5, "Programming": 5},
        extracurriculars={"Coding and robotics": ExtracurricularActivity(name="Coding and robotics", enjoyment=5)},
    )

    recs = recommend_student_profile(onet_data, prof, curated_data=curated_data, limit=8)
    assert len(recs) == 8
    for r in recs:
        assert 0.0 <= r["score"] <= 1.0
        for dim, s in r["dimension_scores"].items():
            assert 0.0 <= s <= 1.0
        assert len(r["why_matched"]) >= 1


def test_empty_profile_recommendations(onet_data, curated_data):
    """Verify recommendation engine handles a completely blank profile safely without crashing."""
    empty_prof = StudentProfile(nickname="BlankStudent")
    recs = recommend_student_profile(onet_data, empty_prof, curated_data=curated_data, limit=5)
    assert len(recs) == 5
    for r in recs:
        assert 0.0 <= r["score"] <= 1.0


def test_recommendation_sensitivity_to_preferences(onet_data, curated_data):
    """Verify recommendations change meaningfully between distinct profiles (e.g. tech student vs medical student)."""
    tech_prof = StudentProfile(
        nickname="TechStudent",
        interests={"Technology and computing": 5},
        subjects={"Computer Science": SubjectPreference(name="Computer Science", interest_level=5, confidence_level=5, academic_performance=95.0)},
        skills={"Programming": 5, "Complex Problem Solving": 5},
        extracurriculars={"Coding and robotics": ExtracurricularActivity(name="Coding and robotics", enjoyment=5)},
    )

    med_prof = StudentProfile(
        nickname="MedStudent",
        interests={"Healthcare and biology": 5},
        subjects={"Biology": SubjectPreference(name="Biology", interest_level=5, confidence_level=5, academic_performance=95.0)},
        skills={"Active Listening": 5, "Social Perceptiveness": 5},
        extracurriculars={"Volunteering and community service": ExtracurricularActivity(name="Volunteering and community service", enjoyment=5)},
    )

    tech_recs = recommend_student_profile(onet_data, tech_prof, curated_data=curated_data, limit=5)
    med_recs = recommend_student_profile(onet_data, med_prof, curated_data=curated_data, limit=5)

    tech_titles = [r["title"].lower() for r in tech_recs]
    med_titles = [r["title"].lower() for r in med_recs]

    # Top recommendations must be differentiated
    assert tech_titles[0] != med_titles[0]
    assert any("program" in t or "developer" in t or "comput" in t or "data" in t or "engineer" in t for t in tech_titles)
    assert any("health" in t or "nurse" in t or "physician" in t or "medical" in t or "therap" in t for t in med_titles)

def test_multiple_selected_interests(onet_data, curated_data):
    """Verify selecting multiple diverse interests blends scoring appropriately."""
    prof = StudentProfile(
        nickname="Polymath",
        interests={"Arts and creative expression": 5, "Technology and computing": 4, "Psychology and understanding people": 4},
    )
    recs = recommend_student_profile(onet_data, prof, curated_data=curated_data, limit=5)
    assert len(recs) == 5
    for r in recs:
        assert 0.0 <= r["score"] <= 1.0
        assert r["active_dimensions"]["interests"] is True


def test_academic_matching_without_marks(onet_data, curated_data):
    """Verify academic matching works properly when performance marks are omitted."""
    prof = StudentProfile(
        nickname="NoMarksStudent",
        subjects={
            "Mathematics": SubjectPreference(name="Mathematics", interest_level=5, confidence_level=4, academic_performance=None),
            "Physics": SubjectPreference(name="Physics", interest_level=4, confidence_level=3, academic_performance=None),
        },
    )
    recs = recommend_student_profile(onet_data, prof, curated_data=curated_data, limit=5)
    assert len(recs) == 5
    for r in recs:
        assert 0.0 <= r["dimension_scores"]["academics"] <= 1.0


def test_no_single_career_monopoly(onet_data, curated_data):
    """Verify recommendations present diverse options, never claiming a single career is the only choice."""
    prof = StudentProfile(nickname="Explorer")
    recs = recommend_student_profile(onet_data, prof, curated_data=curated_data, limit=10)
    titles = set(r["title"] for r in recs)
    assert len(titles) == 10  # Must be 10 distinct occupations


def test_student_recommendations_include_knn_model(onet_data, curated_data):
    """The upgraded recommendation results expose the fitted KNN similarity."""
    prof = StudentProfile(
        nickname="KNNTester",
        stage=EducationalStage.AFTER_CLASS_12.value,
        interests={"Technology and computing": 5, "Science and research": 4},
        subjects={"Mathematics": SubjectPreference(name="Mathematics", interest_level=5, confidence_level=4, academic_performance=90.0)},
        skills={"Programming": 4, "Critical Thinking": 4},
    )
    results = recommend_student_profile(onet_data, prof, curated_data=curated_data, limit=5)
    assert results
    assert all("knn_similarity" in item for item in results)
    assert all(item["scoring_method"].startswith("Hybrid") for item in results)
    assert all(0.0 <= item["score"] <= 1.0 for item in results)
