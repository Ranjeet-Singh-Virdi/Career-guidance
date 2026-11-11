"""Machine Learning Model Definitions, Ensembles, and Inference Pipeline.

Aligned with CSE276: Artificial Intelligence Foundations & SIH 2024 (SIH1781).
Implements a multi-model AI suite:
1. Random Forest Classifier (Bagging Ensemble)
2. Gradient Boosting Classifier (Boosting Ensemble)
3. Multi-Layer Perceptron / Deep Neural Network (MLPClassifier)
4. Support Vector Machine (Calibrated Max-Margin Classifier)
5. K-Nearest Neighbors (Instance-Based Retrieval)
6. Hybrid Soft-Voting AI Ensemble (Fused Meta-Recommender)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

FEATURE_COLUMNS = [
    # Academic Subjects (13)
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
    # RIASEC Holland Codes (6)
    "riasec_realistic",
    "riasec_investigative",
    "riasec_artistic",
    "riasec_social",
    "riasec_enterprising",
    "riasec_conventional",
    # O*NET Core Skills (10)
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

AVAILABLE_MODELS = [
    "Hybrid AI Ensemble (Recommended)",
    "Deep Neural Network (MLP)",
    "Random Forest (Ensemble Bagging)",
    "Gradient Boosting (Boosting Ensemble)",
    "Support Vector Machine (Calibrated SVM)",
]


class AICareerModelSuite:
    """Manages trained AI models, feature transformations, and inference."""

    def __init__(self, model_bundle_path: Path | str | None = None):
        self.model_bundle_path = Path(model_bundle_path or (MODELS_DIR / "career_models.joblib"))
        self.metrics_path = MODELS_DIR / "model_metrics.json"
        self.bundle: dict[str, Any] | None = None
        self.metrics: dict[str, Any] = {}
        self.is_loaded = False
        self._load()

    def _load(self):
        if self.model_bundle_path.exists():
            try:
                self.bundle = joblib.load(self.model_bundle_path)
                self.is_loaded = True
            except Exception as e:
                print(f"Warning: Could not load model bundle: {e}")
                self.bundle = None
                self.is_loaded = False

        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metrics: {e}")
                self.metrics = {}

    def extract_features_from_student_profile(
        self,
        student_profile: Any,
        curated_data: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Extract a standardized 29-dimensional feature vector from a StudentProfile."""
        p_dict = student_profile.to_dict() if hasattr(student_profile, "to_dict") else dict(student_profile)
        subjects = p_dict.get("subjects", {})
        interests = p_dict.get("interests", {})
        skills = p_dict.get("skills", {})

        curated = curated_data or {}
        interest_map = curated.get("interests", {})

        row: dict[str, float] = {}

        # 1. Subject Performance / Interest Mapping
        subj_name_map = {
            "math_score": ["Mathematics", "Maths"],
            "physics_score": ["Physics"],
            "chemistry_score": ["Chemistry"],
            "biology_score": ["Biology"],
            "cs_score": ["Computer Science", "Computing", "Informatics Practices"],
            "accountancy_score": ["Accountancy", "Accounting"],
            "economics_score": ["Economics"],
            "business_studies_score": ["Business Studies", "Business"],
            "history_score": ["History"],
            "english_score": ["English and other languages", "Language & writing", "English"],
            "political_science_score": ["Political Science", "Social studies", "Legal Studies"],
            "psychology_score": ["Psychology"],
            "fine_arts_score": ["Fine Arts", "Arts & design"],
        }

        for col, candidates in subj_name_map.items():
            found_score = None
            for cand in candidates:
                if cand in subjects:
                    s_info = subjects[cand]
                    perf = s_info.get("performance")
                    if perf not in (None, ""):
                        try:
                            found_score = float(perf)
                            break
                        except (ValueError, TypeError):
                            pass
                    # If no explicit percentage marks, infer from interest (1-5) and confidence (1-5)
                    int_lvl = float(s_info.get("interest", 3))
                    conf_lvl = float(s_info.get("confidence", 3))
                    found_score = (int_lvl * 10.0 + conf_lvl * 10.0)  # out of 100
                    break
            row[col] = float(found_score if found_score is not None else 60.0)

        # 2. RIASEC Scores (1.0 - 5.0)
        riasec_keys = {
            "riasec_realistic": "Realistic",
            "riasec_investigative": "Investigative",
            "riasec_artistic": "Artistic",
            "riasec_social": "Social",
            "riasec_enterprising": "Enterprising",
            "riasec_conventional": "Conventional",
        }
        for col, r_name in riasec_keys.items():
            val = None
            if r_name in interests:
                val = float(interests[r_name])
            else:
                # Aggregate from category mappings
                matched = []
                for cat, rating in interests.items():
                    r_tags = interest_map.get(cat, {}).get("onet_riasec", [])
                    if r_name in r_tags:
                        matched.append(float(rating))
                if matched:
                    val = sum(matched) / len(matched)
            row[col] = float(val if val is not None else 2.5)

        # 3. Core Skills (1.0 - 5.0)
        skill_name_map = {
            "skill_critical_thinking": ["Critical Thinking", "Analytical thinking", "Problem-solving"],
            "skill_complex_problem_solving": ["Complex Problem Solving", "Problem-solving", "Analytical thinking"],
            "skill_mathematics": ["Mathematics", "Numerical reasoning"],
            "skill_programming": ["Programming", "Technical skills"],
            "skill_active_listening": ["Active Listening", "Communication", "Empathy"],
            "skill_speaking": ["Speaking", "Communication", "Leadership"],
            "skill_science": ["Science", "Research"],
            "skill_reading_comprehension": ["Reading Comprehension", "Writing"],
            "skill_judgment_decision_making": ["Judgment and Decision Making", "Analytical thinking", "Leadership"],
            "skill_systems_analysis": ["Systems Analysis", "Technical skills", "Attention to detail"],
        }
        for col, candidates in skill_name_map.items():
            found_skill = None
            for cand in candidates:
                for k, v in skills.items():
                    if cand.casefold() in k.casefold() or k.casefold() in cand.casefold():
                        try:
                            found_skill = float(v)
                            break
                        except (ValueError, TypeError):
                            pass
                if found_skill is not None:
                    break
            row[col] = float(found_skill if found_skill is not None else 3.0)

        # Build feature vector matching FEATURE_COLUMNS order
        vec = np.array([[row[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
        return vec

    def predict_careers(
        self,
        features: np.ndarray,
        model_name: str = "Hybrid AI Ensemble (Recommended)",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Predict career rankings and probabilities using the chosen AI model."""
        if not self.is_loaded or self.bundle is None:
            return []

        scaler = self.bundle.get("scaler")
        classes = self.bundle.get("classes", [])
        code_to_career = self.bundle.get("code_to_career", {})
        career_to_code = self.bundle.get("career_to_code", {})
        career_to_family = self.bundle.get("career_to_family", {})

        X_scaled = scaler.transform(features) if scaler is not None else features

        # Generate probabilities based on requested model
        if "Ensemble" in model_name:
            rf = self.bundle.get("random_forest")
            gb = self.bundle.get("gradient_boosting")
            mlp = self.bundle.get("mlp_neural_net")
            svm = self.bundle.get("svm")

            probs_rf = rf.predict_proba(X_scaled)[0] if rf else np.zeros(len(classes))
            probs_gb = gb.predict_proba(X_scaled)[0] if gb else np.zeros(len(classes))
            probs_mlp = mlp.predict_proba(X_scaled)[0] if mlp else np.zeros(len(classes))
            probs_svm = svm.predict_proba(X_scaled)[0] if svm else np.zeros(len(classes))

            # Weighted soft voting
            ensemble_probs = (
                0.30 * probs_rf
                + 0.30 * probs_gb
                + 0.25 * probs_mlp
                + 0.15 * probs_svm
            )
            # Re-normalize
            total = np.sum(ensemble_probs)
            probs = ensemble_probs / total if total > 0 else ensemble_probs

        elif "Neural" in model_name or "MLP" in model_name:
            mlp = self.bundle.get("mlp_neural_net")
            probs = mlp.predict_proba(X_scaled)[0] if mlp else np.zeros(len(classes))

        elif "Random Forest" in model_name:
            rf = self.bundle.get("random_forest")
            probs = rf.predict_proba(X_scaled)[0] if rf else np.zeros(len(classes))

        elif "Gradient Boosting" in model_name:
            gb = self.bundle.get("gradient_boosting")
            probs = gb.predict_proba(X_scaled)[0] if gb else np.zeros(len(classes))

        elif "Support Vector" in model_name or "SVM" in model_name:
            svm = self.bundle.get("svm")
            probs = svm.predict_proba(X_scaled)[0] if svm else np.zeros(len(classes))

        else:
            # Fallback to ensemble
            rf = self.bundle.get("random_forest")
            probs = rf.predict_proba(X_scaled)[0] if rf else np.zeros(len(classes))

        # Sort by predicted probability
        sorted_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(sorted_indices, 1):
            career_title = classes[idx]
            prob = float(probs[idx])
            onet_code = career_to_code.get(career_title, "")
            family = career_to_family.get(career_title, "General Professional")
            results.append({
                "rank": rank,
                "title": career_title,
                "onet_code": onet_code,
                "career_family": family,
                "confidence_score": round(prob, 4),
                "confidence_percent": round(prob * 100.0, 1),
                "model_used": model_name,
            })

        return results

    def get_feature_importances(self) -> dict[str, float]:
        """Return global feature importances from the Random Forest model."""
        if not self.is_loaded or self.bundle is None:
            return {}
        rf = self.bundle.get("random_forest")
        if rf is None or not hasattr(rf, "feature_importances_"):
            return {}
        importances = rf.feature_importances_
        feature_names = self.bundle.get("feature_names", FEATURE_COLUMNS)
        ranking = {name: round(float(imp), 4) for name, imp in zip(feature_names, importances)}
        return dict(sorted(ranking.items(), key=lambda x: x[1], reverse=True))


# Global lazy-loaded singleton
_MODEL_SUITE: AICareerModelSuite | None = None


def get_ai_model_suite() -> AICareerModelSuite:
    global _MODEL_SUITE
    if _MODEL_SUITE is None:
        _MODEL_SUITE = AICareerModelSuite()
    return _MODEL_SUITE
