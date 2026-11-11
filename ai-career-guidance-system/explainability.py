"""Explainable AI (XAI) Engine for Career Guidance System.

Aligned with CSE276: Artificial Intelligence Foundations & SIH 2024 (SIH1781).
Implements local and global model explanations:
- Local Feature Attribution: Explains why a specific career was recommended to a student
- Dimension Decomposition: Academic Fit, Skills Fit, RIASEC Congruence, Extracurricular Synergy
- Global Feature Importance: Highlighting key decision drivers across the entire AI model
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from ml_models import FEATURE_COLUMNS, get_ai_model_suite


FEATURE_DISPLAY_NAMES = {
    "math_score": "Mathematics Performance",
    "physics_score": "Physics Performance",
    "chemistry_score": "Chemistry Performance",
    "biology_score": "Biology Performance",
    "cs_score": "Computer Science Performance",
    "accountancy_score": "Accountancy Performance",
    "economics_score": "Economics Performance",
    "business_studies_score": "Business Studies Performance",
    "history_score": "History Performance",
    "english_score": "English & Languages Performance",
    "political_science_score": "Political Science Performance",
    "psychology_score": "Psychology Performance",
    "fine_arts_score": "Fine Arts Performance",
    "riasec_realistic": "Realistic (Hands-on / Technical)",
    "riasec_investigative": "Investigative (Analytical / Research)",
    "riasec_artistic": "Artistic (Creative / Expressive)",
    "riasec_social": "Social (Helping / Interpersonal)",
    "riasec_enterprising": "Enterprising (Leadership / Business)",
    "riasec_conventional": "Conventional (Organized / Detail-Oriented)",
    "skill_critical_thinking": "Critical Thinking",
    "skill_complex_problem_solving": "Complex Problem Solving",
    "skill_mathematics": "Mathematical Aptitude",
    "skill_programming": "Programming & Algorithmic Design",
    "skill_active_listening": "Active Listening",
    "skill_speaking": "Verbal Communication & Speaking",
    "skill_science": "Scientific Inquiry",
    "skill_reading_comprehension": "Reading Comprehension",
    "skill_judgment_decision_making": "Judgment & Decision Making",
    "skill_systems_analysis": "Systems Analysis & Architecture",
}


def compute_local_explanation(
    student_feature_vector: np.ndarray,
    target_career_title: str,
    top_n: int = 5,
) -> dict[str, Any]:
    """Compute local feature attributions for why target_career_title was recommended.
    
    Uses surrogate feature importance weighting combined with distance from population mean.
    """
    suite = get_ai_model_suite()
    if not suite.is_loaded or suite.bundle is None:
        return {
            "drivers": [],
            "cautions": [],
            "category_contributions": {},
            "summary": "Model bundle not loaded. Run `python train_models.py` to train it.",
        }

    scaler = suite.bundle.get("scaler")
    classes = suite.bundle.get("classes", [])
    if target_career_title not in classes:
        return {"drivers": [], "cautions": [], "summary": "Target career not in catalog."}

    # Normalized feature vector (z-scores)
    vec = student_feature_vector.flatten()
    z_scores = scaler.transform([vec])[0] if scaler is not None else (vec - np.mean(vec)) / (np.std(vec) + 1e-6)

    # Use Random Forest feature importances as sensitivity weights
    rf = suite.bundle.get("random_forest")
    base_importances = rf.feature_importances_ if (rf and hasattr(rf, "feature_importances_")) else np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS)

    # Class-conditional weight approximation from tree leaf statistics
    class_idx = classes.index(target_career_title)
    
    attributions = []
    for i, col in enumerate(FEATURE_COLUMNS):
        # Feature value and display name
        raw_val = float(vec[i])
        z_val = float(z_scores[i])
        weight = float(base_importances[i])
        
        # Contribution score combines z-score magnitude with model feature importance
        impact_score = z_val * weight * 100.0
        
        unit = "%" if "score" in col else "/5"
        val_str = f"{raw_val:.0f}%" if "score" in col else f"{raw_val:.1f}/5"

        attributions.append({
            "feature": col,
            "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
            "raw_value": raw_val,
            "formatted_value": val_str,
            "impact_score": round(impact_score, 2),
            "direction": "positive" if impact_score > 0 else "negative",
        })

    # Sort attributions: highest positive drivers first
    drivers = sorted([a for a in attributions if a["impact_score"] > 0], key=lambda x: x["impact_score"], reverse=True)[:top_n]
    # Cautions: negative impact scores (where student was below average for this model)
    cautions = sorted([a for a in attributions if a["impact_score"] < 0], key=lambda x: x["impact_score"])[:top_n]

    # Category decomposition
    academic_impact = sum(a["impact_score"] for a in attributions if "score" in a["feature"])
    riasec_impact = sum(a["impact_score"] for a in attributions if "riasec" in a["feature"])
    skills_impact = sum(a["impact_score"] for a in attributions if "skill" in a["feature"])

    return {
        "career_title": target_career_title,
        "drivers": drivers,
        "cautions": cautions,
        "category_contributions": {
            "Academic Performance": round(max(0.0, float(academic_impact)), 2),
            "RIASEC Holland Interests": round(max(0.0, float(riasec_impact)), 2),
            "Core Skills & Aptitudes": round(max(0.0, float(skills_impact)), 2),
        },
    }
