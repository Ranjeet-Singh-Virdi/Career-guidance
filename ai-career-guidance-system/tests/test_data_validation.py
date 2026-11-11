"""Automated tests for O*NET schemas, JSON catalogue validation, and missing file resilience."""
from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

from data_loader import (
    DATA_DIR,
    ONET_DIR,
    REQUIRED_ONET_COLUMNS,
    load_all_curated_data,
    load_demo_profiles,
    load_onet_dataset,
    read_csv_safe,
    read_json_safe,
)


def test_onet_required_columns():
    """Verify primary O*NET tables contain the required columns."""
    occ_df = read_csv_safe(ONET_DIR / "occupation_data.csv", REQUIRED_ONET_COLUMNS["occupations"])
    assert not occ_df.empty
    assert "O*NET-SOC Code" in occ_df.columns
    assert "Title" in occ_df.columns

    skills_df = read_csv_safe(ONET_DIR / "essential_skills.csv", REQUIRED_ONET_COLUMNS["skills"], optional=True)
    if not skills_df.empty:
        assert REQUIRED_ONET_COLUMNS["skills"].issubset(set(skills_df.columns))

    interests_df = read_csv_safe(ONET_DIR / "career_interest_types.csv", REQUIRED_ONET_COLUMNS["interests"], optional=True)
    if not interests_df.empty:
        assert REQUIRED_ONET_COLUMNS["interests"].issubset(set(interests_df.columns))


def test_missing_optional_file_handling(tmp_path):
    """Verify that optional missing CSV files return empty DataFrame gracefully without raising FileNotFoundError."""
    non_existent = tmp_path / "optional_file.csv"
    res = read_csv_safe(non_existent, optional=True)
    assert isinstance(res, pd.DataFrame)
    assert res.empty


def test_missing_required_file_raises(tmp_path):
    """Verify that required missing CSV files raise FileNotFoundError."""
    non_existent = tmp_path / "mandatory_file.csv"
    with pytest.raises(FileNotFoundError):
        read_csv_safe(non_existent, optional=False)


def test_missing_required_column_raises(tmp_path):
    """Verify that CSV files missing required columns raise ValueError."""
    dummy_file = tmp_path / "corrupt.csv"
    dummy_file.write_text("ColumnA,ColumnB\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        read_csv_safe(dummy_file, required_cols={"ColumnA", "MissingColumn"})


def test_read_json_safe_handles_missing_and_corrupt(tmp_path):
    """Verify safe JSON loader returns specified default for missing or corrupt files."""
    missing = tmp_path / "does_not_exist.json"
    assert read_json_safe(missing, default=[]) == []

    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{ unclosed json: ", encoding="utf-8")
    assert read_json_safe(corrupt, default={"status": "fallback"}) == {"status": "fallback"}


def test_curated_data_integrity():
    """Verify that all curated JSON files exist and have the expected structure."""
    curated = load_all_curated_data()
    assert isinstance(curated, dict)
    for key in ["careers", "pathways", "subjects", "interests", "extracurriculars"]:
        assert key in curated

    # Check careers structure
    careers = curated["careers"]
    assert len(careers) >= 10
    for c in careers[:5]:
        assert "career_id" in c
        assert "title" in c
        assert "career_family" in c
        assert "possible_education_pathways" in c
        assert "after_class_10" in c["possible_education_pathways"]
        assert "after_class_12" in c["possible_education_pathways"]
        assert "after_graduation" in c["possible_education_pathways"]

    # Check pathways structure
    pathways = curated["pathways"]
    assert "class_10" in pathways
    assert "class_12" in pathways
    assert "graduate" in pathways
    assert len(pathways["class_10"]) >= 4
    assert len(pathways["class_12"]) >= 5
    assert len(pathways["graduate"]) >= 3


def test_demo_profiles_data():
    """Verify demo_profiles.csv loads correctly and contains representative profiles."""
    demos = load_demo_profiles()
    assert not demos.empty
    assert "display_name" in demos.columns
    assert "stage" in demos.columns
    assert len(demos) >= 3
