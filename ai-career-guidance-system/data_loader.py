"""Data loaders for O*NET datasets, curated career catalogues, and educational pathways."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
    _CACHE_DECORATOR = st.cache_data(show_spinner=False)
except Exception:
    def _CACHE_DECORATOR(func):
        return func


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
ONET_DIR = DATA_DIR / "onet"

REQUIRED_ONET_COLUMNS = {
    "occupations": {"O*NET-SOC Code", "Title"},
    "skills": {"O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"},
    "interests": {"O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"},
}


@dataclass
class OnetData:
    occupations: pd.DataFrame
    skills: pd.DataFrame
    interests: pd.DataFrame
    zones: pd.DataFrame
    knowledge: pd.DataFrame
    education: pd.DataFrame
    education_categories: pd.DataFrame
    zone_reference: pd.DataFrame
    software: pd.DataFrame


def read_csv_safe(path: Path | str, required_cols: set[str] | None = None, optional: bool = False) -> pd.DataFrame:
    """Read a CSV file safely with encoding handling and schema checks."""
    file_path = Path(path)
    if not file_path.is_file():
        if optional:
            return pd.DataFrame()
        raise FileNotFoundError(f"Required data file missing: {file_path.name}")
    
    try:
        frame = pd.read_csv(file_path, dtype=str, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    except UnicodeDecodeError:
        frame = pd.read_csv(file_path, dtype=str, encoding="latin1", keep_default_na=False, low_memory=False)
    
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    
    if required_cols and not required_cols.issubset(set(frame.columns)):
        missing = sorted(required_cols - set(frame.columns))
        raise ValueError(f"{file_path.name} is missing required column(s): {', '.join(missing)}")
    
    return frame


def load_onet_dataset(folder: str | Path | None = None) -> OnetData:
    """Load O*NET CSV tables with graceful fallback for missing optional tables."""
    root = Path(folder) if folder else ONET_DIR
    frames = {
        "occupations": read_csv_safe(root / "occupation_data.csv", REQUIRED_ONET_COLUMNS["occupations"]),
        "skills": read_csv_safe(root / "essential_skills.csv", REQUIRED_ONET_COLUMNS["skills"], optional=True),
        "interests": read_csv_safe(root / "career_interest_types.csv", REQUIRED_ONET_COLUMNS["interests"], optional=True),
        "zones": read_csv_safe(root / "job_zones.csv", optional=True),
        "knowledge": read_csv_safe(root / "knowledge.csv", optional=True),
        "education": read_csv_safe(root / "education.csv", optional=True),
        "education_categories": read_csv_safe(root / "education_categories.csv", optional=True),
        "zone_reference": read_csv_safe(root / "job_zone_reference.csv", optional=True),
        "software": read_csv_safe(root / "software_skills.csv", optional=True),
    }
    if not frames["occupations"].empty and "O*NET-SOC Code" in frames["occupations"].columns:
        frames["occupations"] = frames["occupations"].drop_duplicates("O*NET-SOC Code").reset_index(drop=True)
    return OnetData(**frames)


def read_json_safe(path: Path | str, default: Any = None) -> Any:
    """Safely load and parse JSON files, returning default on error or missing file."""
    file_path = Path(path)
    if not file_path.is_file():
        return default if default is not None else {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


@_CACHE_DECORATOR
def load_all_curated_data() -> dict[str, Any]:
    """Load all curated JSON configuration, pathways, and mappings."""
    careers = read_json_safe(DATA_DIR / "careers.json", default=[])
    pathways = read_json_safe(DATA_DIR / "pathways.json", default={})
    subject_mappings = read_json_safe(DATA_DIR / "subject_mappings.json", default={})
    interest_mappings = read_json_safe(DATA_DIR / "interest_mappings.json", default={})
    extracurricular_mappings = read_json_safe(DATA_DIR / "extracurricular_mappings.json", default={})

    return {
        "careers": careers,
        "pathways": pathways,
        "subjects": subject_mappings,
        "interests": interest_mappings,
        "extracurriculars": extracurricular_mappings,
    }


@_CACHE_DECORATOR
def load_demo_profiles(path: str | Path | None = None) -> pd.DataFrame:
    """Load synthetic demo profiles for quick testing and student onboarding."""
    file_path = Path(path) if path else (DATA_DIR / "demo_profiles.csv")
    if not file_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
