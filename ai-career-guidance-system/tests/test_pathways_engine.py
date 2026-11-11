"""Automated tests for pathways_engine and skill gap roadmap generation."""
from __future__ import annotations

import pytest

from assessment import EducationalStage, StudentProfile
from data_loader import load_all_curated_data
from pathways_engine import calculate_skill_gap, generate_learning_roadmap, get_stage_pathways


@pytest.fixture(scope="module")
def curated_data():
    return load_all_curated_data()


def test_class10_pathways(curated_data):
    """Verify Class 10 pathway retrieval includes appropriate academic streams and diplomas."""
    top_careers = [
        {"career_family": "Information Technology", "title": "Software Developer"},
        {"career_family": "Mechanical & Aerospace Engineering", "title": "Mechanical Engineer"},
    ]
    pathways = get_stage_pathways(EducationalStage.AFTER_CLASS_10.value, top_careers, curated_data)

    assert len(pathways) >= 4
    pw_titles = [p["title"] for p in pathways]
    assert any("Science Stream with Mathematics" in t for t in pw_titles)
    assert any("Polytechnic" in t for t in pw_titles)
    for p in pathways:
        assert "typical_subjects" in p
        assert "eligibility_prerequisites" in p


def test_class12_pathways(curated_data):
    """Verify Class 12 pathway retrieval includes degree programs and entrance examinations."""
    top_careers = [
        {"career_family": "Medicine & Healthcare", "title": "General Physician"},
        {"career_family": "Nursing & Allied Health", "title": "Registered Nurse"},
    ]
    pathways = get_stage_pathways(EducationalStage.AFTER_CLASS_12.value, top_careers, curated_data)

    assert len(pathways) >= 5
    pw_titles = [p["title"] for p in pathways]
    assert any("MBBS" in t or "Medical" in t for t in pw_titles)
    for p in pathways:
        assert "entrance_examinations" in p
        assert "typical_duration" in p


def test_graduate_pathways(curated_data):
    """Verify Graduate pathway retrieval includes master's degrees, civil services, and certifications."""
    top_careers = [
        {"career_family": "Civil Services & Governance", "title": "Civil Services Officer"},
        {"career_family": "Management & Strategy", "title": "Management Consultant"},
    ]
    pathways = get_stage_pathways(EducationalStage.GRADUATE.value, top_careers, curated_data)

    assert len(pathways) >= 3
    pw_titles = [p["title"] for p in pathways]
    assert any("Civil Services" in t or "MBA" in t or "Technical" in t for t in pw_titles)


def test_skill_gap_calculation():
    """Verify constructive skill gap calculation and non-negative categorization."""
    mock_career = {
        "title": "Data Scientist",
        "top_skills": [
            ("Mathematics", {"LV": 5.6, "IM": 4.5}),
            ("Critical Thinking", {"LV": 5.0, "IM": 4.0}),
            ("Active Listening", {"LV": 3.0, "IM": 3.0}),
        ],
    }
    student_skills = {
        "Mathematics": 5,        # High self-rating
        "Critical Thinking": 3,  # Moderate self-rating
        # Active Listening is unrated, should use default neutral baseline
    }

    gaps = calculate_skill_gap(mock_career, student_skills)
    assert len(gaps) == 3
    for g in gaps:
        assert "skill_name" in g
        assert "status" in g
        assert g["status"] in ["Growth Opportunity", "Skill in Progress", "Strong Alignment"]
        assert "action_tip" in g


def test_learning_roadmap_generation():
    """Verify 4-week structured roadmap generation with beginner project."""
    mock_career = {
        "title": "Software Developer",
        "curated": {
            "suggested_beginner_projects": ["Build a student portfolio website in HTML/JS"],
        },
    }
    skill_gaps = [
        {"skill_name": "Programming", "gap_difference": 1.5, "status": "Growth Opportunity"},
        {"skill_name": "Complex Problem Solving", "gap_difference": 0.5, "status": "Skill in Progress"},
    ]

    roadmap = generate_learning_roadmap(mock_career, skill_gaps)
    assert len(roadmap["weekly_schedule"]) == 4
    assert roadmap["focus_skill"] == "Programming"
    assert "portfolio website" in roadmap["primary_project"]
    assert len(roadmap["curated_resources"]) >= 4
