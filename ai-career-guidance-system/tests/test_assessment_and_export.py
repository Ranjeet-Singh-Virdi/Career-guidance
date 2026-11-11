"""Automated tests for student profile modeling, demo parsing, comparison logic, and report exports."""
from __future__ import annotations

import pandas as pd
import pytest

from assessment import (
    EducationalStage,
    ExtracurricularActivity,
    StudentProfile,
    SubjectPreference,
    WorkPreferences,
    CareerPriorities,
    create_profile_from_demo,
)
from career_engine import load_onet_data, recommend_student_profile
from data_loader import ONET_DIR, load_all_curated_data, load_demo_profiles


@pytest.fixture(scope="module")
def onet_data():
    return load_onet_data(ONET_DIR)


@pytest.fixture(scope="module")
def curated_data():
    return load_all_curated_data()


def test_educational_stage_enum():
    """Verify EducationalStage enum values and display mappings."""
    stages = [s.value for s in EducationalStage]
    assert "after_class_10" in stages
    assert "after_class_12" in stages
    assert "undergraduate" in stages
    assert "graduate" in stages

    names = EducationalStage.display_names()
    assert len(names) == 4
    for s in stages:
        assert s in names


def test_student_profile_to_dict():
    """Verify StudentProfile serializes completely to dictionary."""
    prof = StudentProfile(
        nickname="Rhea",
        stage=EducationalStage.AFTER_CLASS_10.value,
        academic_stream="Secondary",
        interests={"Technology and computing": 5},
        subjects={"Mathematics": SubjectPreference(name="Mathematics", interest_level=5, confidence_level=4, academic_performance=90.0)},
        skills={"Programming": 4},
        extracurriculars={"Coding and robotics": ExtracurricularActivity(name="Coding and robotics", enjoyment=5)},
    )

    d = prof.to_dict()
    assert d["nickname"] == "Rhea"
    assert d["stage"] == "after_class_10"
    assert "Technology and computing" in d["interests"]
    assert "Mathematics" in d["subjects"]
    assert d["subjects"]["Mathematics"]["performance"] == 90.0
    assert "work_preferences" in d
    assert "career_priorities" in d


def test_create_profile_from_demo_rows():
    """Verify create_profile_from_demo correctly populates StudentProfile from each demo row."""
    demos = load_demo_profiles()
    for _, row in demos.iterrows():
        prof = create_profile_from_demo(row)
        assert isinstance(prof, StudentProfile)
        assert prof.nickname != ""
        assert prof.stage in [s.value for s in EducationalStage]
        assert isinstance(prof.interests, dict)
        assert isinstance(prof.subjects, dict)
        assert isinstance(prof.skills, dict)


def test_career_comparison_structure(onet_data, curated_data):
    """Verify side-by-side career comparison data structures."""
    prof = StudentProfile(nickname="CompTester", interests={"Technology and computing": 5})
    recs = recommend_student_profile(onet_data, prof, curated_data=curated_data, limit=4)
    assert len(recs) >= 3

    # Select 3 careers to compare
    chosen = recs[:3]
    cat_keys = ["interests", "academics", "skills", "extracurriculars", "work_preferences", "career_priorities"]

    for r in chosen:
        assert "dimension_scores" in r
        for k in cat_keys:
            assert k in r["dimension_scores"]
            score = r["dimension_scores"][k]
            assert 0.0 <= score <= 1.0


def test_csv_and_text_report_generation(onet_data, curated_data):
    """Verify report export generation produces valid CSV and comprehensive text."""
    prof = StudentProfile(
        nickname="ExportTester",
        stage=EducationalStage.AFTER_CLASS_12.value,
        interests={"Science and research": 5},
        skills={"Critical Thinking": 4},
    )
    recs = recommend_student_profile(onet_data, prof, curated_data=curated_data, limit=5)

    # Test CSV format
    csv_rows = []
    for r in recs:
        csv_rows.append({
            "Rank": len(csv_rows) + 1,
            "Title": r["title"],
            "Family": r.get("career_family", "General"),
            "Alignment_Score": f"{r['score']:.1%}",
            "ONET_Code": r["code"],
        })
    df_export = pd.DataFrame(csv_rows)
    assert len(df_export) == 5
    assert "Rank" in df_export.columns
    assert "Title" in df_export.columns

    # Test Text Report format
    report_text = f"PATHFINDER REPORT FOR {prof.nickname}\nStage: {prof.stage}\n"
    for idx, r in enumerate(recs, 1):
        report_text += f"\n#{idx} {r['title']} - Score: {r['score']:.1%}\n"

    assert "ExportTester" in report_text
    assert "after_class_12" in report_text
    assert len(report_text) > 100
