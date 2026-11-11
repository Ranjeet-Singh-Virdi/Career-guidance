"""Synthetic Student-Profile Dataset Generator for AI Career Guidance System.

Aligned with CSE276: Artificial Intelligence Foundations & SIH 2024 (SIH1781).
Generates realistic student profiles based on academic subjects, O*NET core skills,
RIASEC interest dimensions, extracurricular activities, and career priorities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"

SUBJECT_COLUMNS = [
    "math_score",
    "physics_score",
    "chemistry_score",
    "biology_score",
    "cs_score",
    "accountancy_score",
    "economics_score",
    "business_studies_score",
    "history_score",
    "english_score",
    "political_science_score",
    "psychology_score",
    "fine_arts_score",
]

RIASEC_COLUMNS = [
    "riasec_realistic",
    "riasec_investigative",
    "riasec_artistic",
    "riasec_social",
    "riasec_enterprising",
    "riasec_conventional",
]

SKILL_COLUMNS = [
    "skill_critical_thinking",
    "skill_complex_problem_solving",
    "skill_mathematics",
    "skill_programming",
    "skill_active_listening",
    "skill_speaking",
    "skill_science",
    "skill_reading_comprehension",
    "skill_judgment_decision_making",
    "skill_systems_analysis",
]

FEATURE_COLUMNS = SUBJECT_COLUMNS + RIASEC_COLUMNS + SKILL_COLUMNS


def _load_career_catalog() -> list[dict[str, Any]]:
    catalog_path = DATA_DIR / "careers.json"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Careers catalog missing at {catalog_path}")
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def generate_synthetic_student_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Generate a balanced, realistic synthetic student-profile dataset.
    
    Each record represents an individual student profile with:
    - Academic performance percentages (0-100) across foundational subjects
    - Holland RIASEC interest scores (1.0 - 5.0)
    - O*NET core foundational skill proficiencies (1.0 - 5.0)
    - Educational stage & career priorities
    - Ground truth target career label (ONET code & title)
    """
    rng = np.random.default_rng(random_state)
    careers = _load_career_catalog()
    num_careers = len(careers)

    # Base archetypes corresponding to each curated career in careers.json
    # Each archetype defines the natural centroid around which students gravitate.
    archetypes: list[dict[str, Any]] = []
    for c in careers:
        title = c["title"]
        family = c.get("career_family", "")
        code = c["onet_code"]
        
        # Archetype defaults
        subj_means = {s: 60.0 for s in SUBJECT_COLUMNS}
        riasec_means = {r: 2.5 for r in RIASEC_COLUMNS}
        skill_means = {sk: 3.0 for sk in SKILL_COLUMNS}

        # Tailor archetype means based on career domain
        if "Software" in title or "Information Security" in title or "Cybersecurity" in title:
            subj_means["math_score"] = 85.0
            subj_means["cs_score"] = 90.0
            subj_means["physics_score"] = 78.0
            riasec_means["riasec_investigative"] = 4.5
            riasec_means["riasec_realistic"] = 3.8
            riasec_means["riasec_conventional"] = 3.6
            skill_means["skill_programming"] = 4.5
            skill_means["skill_complex_problem_solving"] = 4.3
            skill_means["skill_critical_thinking"] = 4.2
            skill_means["skill_mathematics"] = 4.0

        elif "Data Scientist" in title:
            subj_means["math_score"] = 92.0
            subj_means["cs_score"] = 88.0
            subj_means["economics_score"] = 75.0
            subj_means["physics_score"] = 80.0
            riasec_means["riasec_investigative"] = 4.8
            riasec_means["riasec_conventional"] = 3.8
            skill_means["skill_mathematics"] = 4.8
            skill_means["skill_critical_thinking"] = 4.6
            skill_means["skill_programming"] = 4.2
            skill_means["skill_complex_problem_solving"] = 4.4

        elif "Mechanical" in title or "Robotics" in title:
            subj_means["physics_score"] = 88.0
            subj_means["math_score"] = 85.0
            subj_means["chemistry_score"] = 75.0
            riasec_means["riasec_realistic"] = 4.7
            riasec_means["riasec_investigative"] = 4.2
            skill_means["skill_complex_problem_solving"] = 4.3
            skill_means["skill_mathematics"] = 4.2
            skill_means["skill_systems_analysis"] = 4.0

        elif "Civil" in title:
            subj_means["physics_score"] = 84.0
            subj_means["math_score"] = 82.0
            subj_means["chemistry_score"] = 74.0
            riasec_means["riasec_realistic"] = 4.6
            riasec_means["riasec_conventional"] = 3.5
            skill_means["skill_complex_problem_solving"] = 4.1
            skill_means["skill_mathematics"] = 4.0
            skill_means["skill_critical_thinking"] = 3.9

        elif "Electrical" in title:
            subj_means["physics_score"] = 89.0
            subj_means["math_score"] = 87.0
            subj_means["cs_score"] = 78.0
            riasec_means["riasec_realistic"] = 4.5
            riasec_means["riasec_investigative"] = 4.3
            skill_means["skill_systems_analysis"] = 4.3
            skill_means["skill_complex_problem_solving"] = 4.2
            skill_means["skill_mathematics"] = 4.4

        elif "Physician" in title or "Doctor" in title:
            subj_means["biology_score"] = 92.0
            subj_means["chemistry_score"] = 88.0
            subj_means["physics_score"] = 80.0
            riasec_means["riasec_investigative"] = 4.6
            riasec_means["riasec_social"] = 4.5
            skill_means["skill_science"] = 4.8
            skill_means["skill_active_listening"] = 4.6
            skill_means["skill_critical_thinking"] = 4.5
            skill_means["skill_judgment_decision_making"] = 4.4

        elif "Nurse" in title:
            subj_means["biology_score"] = 85.0
            subj_means["chemistry_score"] = 78.0
            subj_means["psychology_score"] = 82.0
            riasec_means["riasec_social"] = 4.8
            riasec_means["riasec_investigative"] = 3.8
            skill_means["skill_active_listening"] = 4.8
            skill_means["skill_speaking"] = 4.4
            skill_means["skill_science"] = 3.9

        elif "Pharmacist" in title:
            subj_means["chemistry_score"] = 90.0
            subj_means["biology_score"] = 84.0
            subj_means["math_score"] = 76.0
            riasec_means["riasec_investigative"] = 4.4
            riasec_means["riasec_conventional"] = 4.0
            skill_means["skill_science"] = 4.5
            skill_means["skill_critical_thinking"] = 4.2
            skill_means["skill_reading_comprehension"] = 4.3

        elif "Biotechnologist" in title:
            subj_means["biology_score"] = 90.0
            subj_means["chemistry_score"] = 86.0
            subj_means["cs_score"] = 76.0
            riasec_means["riasec_investigative"] = 4.7
            riasec_means["riasec_realistic"] = 3.6
            skill_means["skill_science"] = 4.7
            skill_means["skill_critical_thinking"] = 4.4
            skill_means["skill_complex_problem_solving"] = 4.2

        elif "Accountant" in title or "Auditor" in title:
            subj_means["accountancy_score"] = 92.0
            subj_means["economics_score"] = 86.0
            subj_means["math_score"] = 82.0
            subj_means["business_studies_score"] = 84.0
            riasec_means["riasec_conventional"] = 4.8
            riasec_means["riasec_enterprising"] = 3.8
            skill_means["skill_mathematics"] = 4.4
            skill_means["skill_critical_thinking"] = 4.2
            skill_means["skill_judgment_decision_making"] = 4.1

        elif "Investment Banker" in title or "Financial Analyst" in title:
            subj_means["economics_score"] = 90.0
            subj_means["math_score"] = 88.0
            subj_means["accountancy_score"] = 86.0
            subj_means["business_studies_score"] = 85.0
            riasec_means["riasec_enterprising"] = 4.6
            riasec_means["riasec_conventional"] = 4.3
            riasec_means["riasec_investigative"] = 4.0
            skill_means["skill_critical_thinking"] = 4.5
            skill_means["skill_judgment_decision_making"] = 4.4
            skill_means["skill_mathematics"] = 4.3

        elif "Management Consultant" in title:
            subj_means["business_studies_score"] = 90.0
            subj_means["economics_score"] = 86.0
            subj_means["english_score"] = 88.0
            riasec_means["riasec_enterprising"] = 4.8
            riasec_means["riasec_social"] = 3.9
            skill_means["skill_speaking"] = 4.7
            skill_means["skill_critical_thinking"] = 4.6
            skill_means["skill_complex_problem_solving"] = 4.5
            skill_means["skill_leadership"] = 4.5

        elif "Lawyer" in title or "Advocate" in title:
            subj_means["political_science_score"] = 90.0
            subj_means["history_score"] = 88.0
            subj_means["english_score"] = 92.0
            riasec_means["riasec_enterprising"] = 4.5
            riasec_means["riasec_investigative"] = 4.3
            riasec_means["riasec_social"] = 3.8
            skill_means["skill_critical_thinking"] = 4.8
            skill_means["skill_speaking"] = 4.7
            skill_means["skill_reading_comprehension"] = 4.8
            skill_means["skill_active_listening"] = 4.4

        elif "Civil Services" in title:
            subj_means["history_score"] = 90.0
            subj_means["political_science_score"] = 92.0
            subj_means["economics_score"] = 86.0
            subj_means["english_score"] = 88.0
            riasec_means["riasec_enterprising"] = 4.4
            riasec_means["riasec_social"] = 4.5
            riasec_means["riasec_investigative"] = 4.2
            skill_means["skill_judgment_decision_making"] = 4.7
            skill_means["skill_critical_thinking"] = 4.6
            skill_means["skill_speaking"] = 4.5
            skill_means["skill_reading_comprehension"] = 4.6

        elif "Psychologist" in title:
            subj_means["psychology_score"] = 94.0
            subj_means["biology_score"] = 78.0
            subj_means["english_score"] = 86.0
            riasec_means["riasec_social"] = 4.8
            riasec_means["riasec_investigative"] = 4.4
            skill_means["skill_active_listening"] = 4.9
            skill_means["skill_speaking"] = 4.5
            skill_means["skill_critical_thinking"] = 4.3

        elif "UI/UX" in title or "Product Designer" in title:
            subj_means["fine_arts_score"] = 90.0
            subj_means["cs_score"] = 82.0
            subj_means["psychology_score"] = 84.0
            riasec_means["riasec_artistic"] = 4.9
            riasec_means["riasec_investigative"] = 3.9
            skill_means["skill_critical_thinking"] = 4.3
            skill_means["skill_active_listening"] = 4.2
            skill_means["skill_complex_problem_solving"] = 4.1

        elif "Architect" in title:
            subj_means["math_score"] = 86.0
            subj_means["physics_score"] = 84.0
            subj_means["fine_arts_score"] = 88.0
            riasec_means["riasec_artistic"] = 4.6
            riasec_means["riasec_realistic"] = 4.3
            skill_means["skill_complex_problem_solving"] = 4.4
            skill_means["skill_critical_thinking"] = 4.2
            skill_means["skill_mathematics"] = 4.2

        elif "Journalist" in title:
            subj_means["english_score"] = 94.0
            subj_means["history_score"] = 86.0
            subj_means["political_science_score"] = 88.0
            riasec_means["riasec_artistic"] = 4.5
            riasec_means["riasec_investigative"] = 4.4
            riasec_means["riasec_enterprising"] = 4.0
            skill_means["skill_reading_comprehension"] = 4.8
            skill_means["skill_speaking"] = 4.7
            skill_means["skill_critical_thinking"] = 4.5

        elif "Agricultural" in title or "Agronomist" in title:
            subj_means["biology_score"] = 88.0
            subj_means["chemistry_score"] = 82.0
            subj_means["physics_score"] = 74.0
            riasec_means["riasec_realistic"] = 4.8
            riasec_means["riasec_investigative"] = 4.1
            skill_means["skill_science"] = 4.5
            skill_means["skill_complex_problem_solving"] = 4.1
            skill_means["skill_critical_thinking"] = 4.0

        archetypes.append({
            "title": title,
            "onet_code": code,
            "career_family": family,
            "subj_means": subj_means,
            "riasec_means": riasec_means,
            "skill_means": skill_means,
        })

    # Sample equally across archetypes with realistic variance
    samples_per_archetype = n_samples // num_careers
    rows = []
    stages = ["after_class_10", "after_class_12", "undergraduate", "graduate"]
    stage_probs = [0.25, 0.40, 0.25, 0.10]

    for arch_idx, arch in enumerate(archetypes):
        target_title = arch["title"]
        target_code = arch["onet_code"]
        target_family = arch["career_family"]

        for _ in range(samples_per_archetype):
            row_data: dict[str, Any] = {
                "student_id": f"STU-{len(rows)+1:06d}",
                "target_career": target_title,
                "target_onet_code": target_code,
                "career_family": target_family,
                "stage": rng.choice(stages, p=stage_probs),
            }

            # Generate subject marks with normal distribution around archetype means
            for subj in SUBJECT_COLUMNS:
                mean = arch["subj_means"][subj]
                # If subject is a key subject for this archetype, std dev is lower, score higher
                std = 8.0 if mean >= 75.0 else 12.0
                score = rng.normal(loc=mean, scale=std)
                # Clip marks to [35.0, 99.0]
                row_data[subj] = round(float(np.clip(score, 35.0, 99.0)), 1)

            # Generate RIASEC scores (1.0 to 5.0)
            for r in RIASEC_COLUMNS:
                mean = arch["riasec_means"][r]
                score = rng.normal(loc=mean, scale=0.55)
                row_data[r] = round(float(np.clip(score, 1.0, 5.0)), 2)

            # Generate Skill scores (1.0 to 5.0)
            for sk in SKILL_COLUMNS:
                mean = arch["skill_means"][sk]
                score = rng.normal(loc=mean, scale=0.5)
                row_data[sk] = round(float(np.clip(score, 1.0, 5.0)), 2)

            rows.append(row_data)

    # If samples didn't divide evenly, add remaining
    while len(rows) < n_samples:
        arch = rng.choice(archetypes)
        row_data = {
            "student_id": f"STU-{len(rows)+1:06d}",
            "target_career": arch["title"],
            "target_onet_code": arch["onet_code"],
            "career_family": arch["career_family"],
            "stage": rng.choice(stages, p=stage_probs),
        }
        for subj in SUBJECT_COLUMNS:
            row_data[subj] = round(float(np.clip(rng.normal(arch["subj_means"][subj], 10.0), 35.0, 99.0)), 1)
        for r in RIASEC_COLUMNS:
            row_data[r] = round(float(np.clip(rng.normal(arch["riasec_means"][r], 0.5), 1.0, 5.0)), 2)
        for sk in SKILL_COLUMNS:
            row_data[sk] = round(float(np.clip(rng.normal(arch["skill_means"][sk], 0.5), 1.0, 5.0)), 2)
        rows.append(row_data)

    df = pd.DataFrame(rows)
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path is None:
        output_path = DATA_DIR / "synthetic_student_profiles.csv"
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} synthetic student profiles saved to {out_file}")
    return df


if __name__ == "__main__":
    generate_synthetic_student_dataset(n_samples=5000)
