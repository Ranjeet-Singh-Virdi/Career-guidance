"""Data preparation and transparent O*NET content-based career matching with 6-dimensional weighted scoring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ml_recommender import recommend_with_knn
from ml_models import get_ai_model_suite, AVAILABLE_MODELS
from explainability import compute_local_explanation

from data_loader import (
    OnetData,
    load_all_curated_data,
    load_onet_dataset,
    read_csv_safe,
    REQUIRED_ONET_COLUMNS,
)

INTERESTS = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]

SUBJECT_KNOWLEDGE = {
    "Mathematics": ["Mathematics", "Physics", "Economics and Accounting"],
    "Science": ["Biology", "Chemistry", "Physics", "Medicine and Dentistry", "Engineering and Technology"],
    "Language & writing": ["English Language", "Communications and Media", "Foreign Language", "History and Archeology"],
    "Computing": ["Computers and Electronics", "Telecommunications", "Mathematics"],
    "Business": ["Administration and Management", "Economics and Accounting", "Sales and Marketing", "Personnel and Human Resources"],
    "Arts & design": ["Design", "Fine Arts", "Communications and Media", "English Language"],
    "Social studies": ["Psychology", "Sociology and Anthropology", "Law and Government", "Education and Training", "Philosophy and Theology"],
    "Physics": ["Physics", "Engineering and Technology", "Mathematics"],
    "Chemistry": ["Chemistry", "Biology", "Medicine and Dentistry"],
    "Biology": ["Biology", "Medicine and Dentistry", "Chemistry"],
    "Computer Science": ["Computers and Electronics", "Telecommunications", "Mathematics"],
    "English and other languages": ["English Language", "Communications and Media", "Foreign Language"],
    "History": ["History and Archeology", "Sociology and Anthropology"],
    "Geography": ["Geography", "History and Archeology"],
    "Political Science": ["Law and Government", "History and Archeology"],
    "Economics": ["Economics and Accounting", "Mathematics"],
    "Accountancy": ["Economics and Accounting", "Administration and Management"],
    "Business Studies": ["Administration and Management", "Sales and Marketing"],
    "Psychology": ["Psychology", "Therapy and Counseling"],
    "Sociology": ["Sociology and Anthropology", "Philosophy and Theology"],
    "Fine Arts": ["Fine Arts", "Design"],
    "Physical Education": ["Medicine and Dentistry", "Education and Training"],
    "Agriculture": ["Food Production", "Biology", "Chemistry"],
    "Legal Studies": ["Law and Government", "Philosophy and Theology"],
}

DEFAULT_WEIGHTS = {
    "interests": 0.25,
    "academics": 0.20,
    "skills": 0.20,
    "extracurriculars": 0.15,
    "work_preferences": 0.10,
    "career_priorities": 0.10,
}

REQUIRED = REQUIRED_ONET_COLUMNS


def load_onet_data(folder: str | Path | None = None) -> OnetData:
    """Backward-compatible loader wrapping data_loader.load_onet_dataset."""
    return load_onet_dataset(folder)


def _scale_map(frame: pd.DataFrame, scales: set[str], nested: bool = False) -> dict:
    """Convert long O*NET ratings table into fast in-memory lookups."""
    required = {"O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"}
    if frame.empty or not required.issubset(set(frame.columns)):
        return {}
    selected = frame[frame["Scale ID"].astype(str).str.upper().isin(scales)]
    numeric = pd.to_numeric(selected["Data Value"], errors="coerce")
    result: dict = {}
    for code, name, scale, value in zip(
        selected["O*NET-SOC Code"], selected["Element Name"], selected["Scale ID"], numeric
    ):
        if not code or not name or pd.isna(value):
            continue
        code_str, name_str, scale_str = str(code), str(name), str(scale).upper()
        if nested:
            result.setdefault(code_str, {}).setdefault(name_str, {})[scale_str] = float(value)
        else:
            result.setdefault(code_str, {})[name_str] = float(value)
    return result


def _education_lookup(data: OnetData) -> dict[str, str]:
    if data.education.empty or not {"O*NET-SOC Code", "Category", "Data Value"}.issubset(data.education.columns):
        return {}
    rows = data.education.copy()
    if "Scale ID" in rows:
        rows = rows[rows["Scale ID"].str.upper() == "RL"]
    rows["_value"] = pd.to_numeric(rows["Data Value"], errors="coerce")
    rows["_category"] = pd.to_numeric(rows["Category"], errors="coerce")
    rows = rows.dropna(subset=["_value", "_category"])
    if rows.empty:
        return {}
    best = rows.sort_values(["O*NET-SOC Code", "_value", "_category"], ascending=[True, False, True]).drop_duplicates("O*NET-SOC Code")
    descriptions: dict[str, str] = {}
    cats = data.education_categories
    if not cats.empty and {"Category", "Category Description"}.issubset(cats.columns):
        if "Scale ID" in cats:
            cats = cats[cats["Scale ID"].str.upper() == "RL"]
        descriptions = dict(zip(cats["Category"].astype(str), cats["Category Description"].astype(str)))
    return {str(row["O*NET-SOC Code"]): descriptions.get(str(int(row["_category"])), f"Education category {int(row['_category'])}") for _, row in best.iterrows()}


def _zone_lookup(data: OnetData) -> tuple[dict[str, str], dict[str, str]]:
    zones: dict[str, str] = {}
    details: dict[str, str] = {}
    if not data.zones.empty and {"O*NET-SOC Code", "Job Zone"}.issubset(data.zones.columns):
        zones = dict(zip(data.zones["O*NET-SOC Code"], data.zones["Job Zone"].astype(str)))
    ref = data.zone_reference
    if not ref.empty and {"Job Zone", "Name", "Education", "Experience", "Job Training"}.issubset(ref.columns):
        details = {str(row["Job Zone"]): f"{row['Name']}. Education: {row['Education']} Experience: {row['Experience']} Training: {row['Job Training']}" for _, row in ref.iterrows()}
    return zones, details


def redistribute_weights(active_dimensions: dict[str, bool], base_weights: dict[str, float] | None = None) -> dict[str, float]:
    """Redistribute weights proportionally when certain dimensions are skipped."""
    weights = dict(base_weights or DEFAULT_WEIGHTS)
    
    active_keys = [k for k, is_active in active_dimensions.items() if is_active and k in weights]
    if not active_keys:
        return {k: 1.0 / len(weights) for k in weights}
    
    total_active_weight = sum(weights[k] for k in active_keys)
    if total_active_weight <= 0:
        return {k: (1.0 / len(active_keys) if k in active_keys else 0.0) for k in weights}
    
    normalized: dict[str, float] = {}
    for k in weights:
        if k in active_keys:
            normalized[k] = weights[k] / total_active_weight
        else:
            normalized[k] = 0.0
            
    return normalized


def recommend(
    data: OnetData,
    selected_interests: list[str],
    ratings: Mapping[str, int],
    subjects: Mapping[str, int] | None = None,
    limit: int = 8,
) -> list[dict]:
    """Preserve existing 3-to-5 argument signature with exact backward compatibility."""
    subjects = subjects or {}
    skills_by_code = _scale_map(data.skills, {"IM", "LV"}, nested=True)
    interest_by_code = _scale_map(data.interests, {"OI"})
    knowledge_by_code = _scale_map(data.knowledge, {"IM"})
    software_by_code: dict[str, list[str]] = {}
    if not data.software.empty and {"O*NET-SOC Code", "Workplace Example"}.issubset(data.software.columns):
        for code, example in zip(data.software["O*NET-SOC Code"], data.software["Workplace Example"]):
            code, example = str(code), str(example)
            examples = software_by_code.setdefault(code, [])
            if example and example not in examples and len(examples) < 5:
                examples.append(example)
    zone_by_code, zone_details = _zone_lookup(data)
    education_by_code = _education_lookup(data)

    skill_ratings = {str(key).casefold(): max(1, min(5, int(value))) / 5 for key, value in ratings.items()}
    active_subjects = {key: max(0, min(100, int(value))) / 100 for key, value in subjects.items() if key in SUBJECT_KNOWLEDGE}

    if active_subjects:
        weights = {"interest": 0.45, "skill": 0.40, "academic": 0.15}
    else:
        weights = {"interest": 0.55, "skill": 0.45, "academic": 0.0}

    results = []
    for _, occupation in data.occupations.iterrows():
        code, title = str(occupation.get("O*NET-SOC Code", "")), str(occupation.get("Title", ""))
        if not code or not title:
            continue
        interest_values = interest_by_code.get(code, {})
        picked_interest_values = [interest_values.get(label) for label in selected_interests]
        interest_values_present = [value for value in picked_interest_values if value is not None]
        interest_fit = sum(interest_values_present) / (len(interest_values_present) * 7) if interest_values_present else 0.5

        skill_values = skills_by_code.get(code, {})
        top_skills = sorted(skill_values.items(), key=lambda item: item[1].get("IM", item[1].get("LV", 0)), reverse=True)[:10]
        if top_skills:
            weighted_scores, importance_weights = [], []
            for name, scales in top_skills:
                importance = max(scales.get("IM", 2.5), 0.5)
                required = scales.get("LV", scales.get("IM", 3.0))
                current = skill_ratings.get(name.casefold(), 0.5)
                weighted_scores.append(min(current / max(required / 7, 0.25), 1.0))
                importance_weights.append(importance)
            skill_fit = sum(score * weight for score, weight in zip(weighted_scores, importance_weights)) / sum(importance_weights)
        else:
            skill_fit = 0.5

        knowledge_values = knowledge_by_code.get(code, {})
        academic_pairs = []
        for subject, score in active_subjects.items():
            for domain in SUBJECT_KNOWLEDGE[subject]:
                if domain in knowledge_values:
                    academic_pairs.append((score, max(knowledge_values[domain], 0.5)))
        if academic_pairs:
            academic_fit = sum(score * demand for score, demand in academic_pairs) / sum(demand for _, demand in academic_pairs)
            academic_known = True
        else:
            academic_fit, academic_known = 0.5, False

        used_weights = dict(weights)
        if active_subjects and not academic_known:
            used_weights["interest"] += used_weights["academic"] * 0.5
            used_weights["skill"] += used_weights["academic"] * 0.5
            used_weights["academic"] = 0.0

        total = used_weights["interest"] * interest_fit + used_weights["skill"] * skill_fit + used_weights["academic"] * academic_fit
        software_examples = software_by_code.get(code, [])
        results.append({
            "code": code,
            "title": title,
            "description": str(occupation.get("Description", "")),
            "score": max(0.0, min(1.0, float(total))),
            "interest": interest_fit,
            "skill_fit": skill_fit,
            "academic": academic_fit,
            "academic_known": academic_known,
            "skills": top_skills,
            "zone": zone_by_code.get(code, "—"),
            "zone_detail": zone_details.get(zone_by_code.get(code, ""), ""),
            "education": education_by_code.get(code, "Not reported in the included O*NET education table"),
            "software": software_examples,
            "weights": used_weights,
        })
    return sorted(results, key=lambda item: (item["score"], item["title"]), reverse=True)[:max(1, int(limit))]


def recommend_student_profile(
    data: OnetData,
    profile: Any,
    curated_data: dict[str, Any] | None = None,
    custom_weights: dict[str, float] | None = None,
    limit: int = 10,
    model_name: str = "Hybrid AI Ensemble (Recommended)",
) -> list[dict[str, Any]]:
    """Complete 6-dimensional recommendation engine with explainable scoring."""
    curated = curated_data or load_all_curated_data()
    careers_catalog: list[dict] = curated.get("careers", [])
    interest_map: dict = curated.get("interests", {})
    subject_map: dict = curated.get("subjects", {})
    extra_map: dict = curated.get("extracurriculars", {})

    p_dict = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)
    student_interests: dict[str, int] = p_dict.get("interests", {})
    student_subjects: dict[str, dict] = p_dict.get("subjects", {})
    student_skills: dict[str, int] = p_dict.get("skills", {})
    student_extras: dict[str, dict] = p_dict.get("extracurriculars", {})
    student_work: dict = p_dict.get("work_preferences", {})
    student_priorities: dict = p_dict.get("career_priorities", {})

    active_dimensions = {
        "interests": bool(student_interests),
        "academics": bool(student_subjects),
        "skills": bool(student_skills),
        "extracurriculars": bool(student_extras),
        "work_preferences": bool(any(v and v != "not_sure" for v in student_work.values())),
        "career_priorities": bool(student_priorities.get("priorities")),
    }

    active_weights = redistribute_weights(active_dimensions, custom_weights)

    skills_by_code = _scale_map(data.skills, {"IM", "LV"}, nested=True)
    interest_by_code = _scale_map(data.interests, {"OI"})
    knowledge_by_code = _scale_map(data.knowledge, {"IM"})
    software_by_code: dict[str, list[str]] = {}
    if not data.software.empty and {"O*NET-SOC Code", "Workplace Example"}.issubset(data.software.columns):
        for code, example in zip(data.software["O*NET-SOC Code"], data.software["Workplace Example"]):
            code_str, ex_str = str(code), str(example)
            ex_list = software_by_code.setdefault(code_str, [])
            if ex_str and ex_str not in ex_list and len(ex_list) < 5:
                ex_list.append(ex_str)
    zone_by_code, zone_details = _zone_lookup(data)
    education_by_code = _education_lookup(data)

    curated_by_code = {c.get("onet_code"): c for c in careers_catalog if c.get("onet_code")}

    student_riasec_weights: dict[str, float] = {}
    for cat_name, rating in student_interests.items():
        weight = float(rating) / 5.0
        cat_info = interest_map.get(cat_name, {})
        for riasec in cat_info.get("onet_riasec", []):
            student_riasec_weights[riasec] = max(student_riasec_weights.get(riasec, 0.0), weight)
    for r in INTERESTS:
        if r in student_interests:
            student_riasec_weights[r] = max(student_riasec_weights.get(r, 0.0), float(student_interests[r]) / 5.0)

    student_extra_skills: set[str] = set()
    student_extra_families: set[str] = set()
    for act_name in student_extras:
        act_info = extra_map.get(act_name, {})
        for sk in act_info.get("transferable_skills", []):
            student_extra_skills.add(sk.casefold())
        for fam in act_info.get("career_families", []):
            student_extra_families.add(fam.casefold())

    results = []
    for _, occupation in data.occupations.iterrows():
        code, title = str(occupation.get("O*NET-SOC Code", "")), str(occupation.get("Title", ""))
        if not code or not title:
            continue

        curated_entry = curated_by_code.get(code)
        career_family = curated_entry.get("career_family", "General Professional") if curated_entry else "General Professional"

        # 1. Interest Alignment Score
        interest_values = interest_by_code.get(code, {})
        if active_dimensions["interests"] and student_riasec_weights:
            riasec_scores = []
            for r_code, student_weight in student_riasec_weights.items():
                if r_code in interest_values:
                    norm_onet = float(interest_values[r_code]) / 7.0
                    riasec_scores.append(norm_onet * student_weight)
            curated_bonus = 0.0
            if curated_entry:
                curated_interests = curated_entry.get("relevant_interests", [])
                matching_curated = sum(1 for i in student_interests if i in curated_interests)
                curated_bonus = min(0.2, matching_curated * 0.08)
            raw_interest = (sum(riasec_scores) / max(len(riasec_scores), 1)) if riasec_scores else 0.5
            interest_score = min(1.0, max(0.0, raw_interest * 0.8 + curated_bonus))
        else:
            interest_score = 0.5

        # 2. Academic Alignment Score
        knowledge_values = knowledge_by_code.get(code, {})
        if active_dimensions["academics"] and student_subjects:
            subject_pair_scores = []
            for subj_name, pref in student_subjects.items():
                int_score = float(pref.get("interest", 3)) / 5.0
                conf_score = float(pref.get("confidence", 3)) / 5.0
                perf_val = pref.get("performance")
                if perf_val is not None and str(perf_val).strip() != "":
                    try:
                        perf_score = min(1.0, max(0.0, float(perf_val) / 100.0))
                        combined_subj = 0.4 * int_score + 0.3 * conf_score + 0.3 * perf_score
                    except ValueError:
                        combined_subj = 0.5 * int_score + 0.5 * conf_score
                else:
                    combined_subj = 0.5 * int_score + 0.5 * conf_score

                domains = subject_map.get(subj_name, {}).get("onet_knowledge", SUBJECT_KNOWLEDGE.get(subj_name, []))
                matched_demands = [knowledge_values[d] for d in domains if d in knowledge_values]
                if matched_demands:
                    avg_demand = sum(matched_demands) / len(matched_demands)
                    subject_pair_scores.append((combined_subj, max(avg_demand, 1.0)))

            if subject_pair_scores:
                academic_score = sum(s * d for s, d in subject_pair_scores) / sum(d for _, d in subject_pair_scores)
            else:
                academic_score = 0.5
        else:
            academic_score = 0.5

        # 3. Skill Alignment Score
        skill_values = skills_by_code.get(code, {})
        top_skills = sorted(skill_values.items(), key=lambda item: item[1].get("IM", item[1].get("LV", 0)), reverse=True)[:10]
        if active_dimensions["skills"] and student_skills and top_skills:
            norm_ratings = {k.casefold(): float(v) / 5.0 for k, v in student_skills.items()}
            weighted_scores, importance_weights = [], []
            for name, scales in top_skills:
                importance = max(scales.get("IM", 2.5), 0.5)
                required = scales.get("LV", scales.get("IM", 3.0))
                current = norm_ratings.get(name.casefold(), 0.5)
                weighted_scores.append(min(current / max(required / 7.0, 0.25), 1.0))
                importance_weights.append(importance)
            skill_score = sum(s * w for s, w in zip(weighted_scores, importance_weights)) / sum(importance_weights)
        else:
            skill_score = 0.5

        # 4. Extracurricular Transferable Skill Score
        if active_dimensions["extracurriculars"] and student_extras:
            skill_matches = sum(1 for name, _ in top_skills if name.casefold() in student_extra_skills)
            extra_skill_ratio = min(1.0, skill_matches / max(len(top_skills), 1) * 2.0)
            family_match = 1.0 if career_family.casefold() in student_extra_families else 0.4
            extracurricular_score = min(1.0, max(0.0, 0.6 * extra_skill_ratio + 0.4 * family_match))
        else:
            extracurricular_score = 0.5

        # 5. Work Preferences Score
        if active_dimensions["work_preferences"]:
            work_score_items = []
            focus_dim = student_work.get("focus_dimension", "ideas")
            if focus_dim == "ideas":
                work_score_items.append(0.85 if "Investigative" in interest_values or "Artistic" in interest_values else 0.5)
            elif focus_dim == "people":
                work_score_items.append(0.9 if "Social" in interest_values or "Enterprising" in interest_values else 0.4)
            elif focus_dim == "data":
                work_score_items.append(0.85 if "Conventional" in interest_values or "Investigative" in interest_values else 0.5)
            elif focus_dim == "physical_objects":
                work_score_items.append(0.9 if "Realistic" in interest_values else 0.4)

            env_type = student_work.get("environment_type", "mixed")
            zone_num = int(zone_by_code.get(code, "3")) if zone_by_code.get(code, "").isdigit() else 3
            if env_type == "office" and zone_num >= 3:
                work_score_items.append(0.8)
            elif env_type == "laboratory" and ("Science" in career_family or "Health" in career_family or "Bio" in career_family):
                work_score_items.append(0.9)
            elif env_type == "workshop" and ("Engineering" in career_family or "Trades" in career_family):
                work_score_items.append(0.9)
            else:
                work_score_items.append(0.6)

            work_preference_score = sum(work_score_items) / max(len(work_score_items), 1)
        else:
            work_preference_score = 0.5

        # 6. Career Priorities Score
        priorities_list = student_priorities.get("priorities", [])
        if active_dimensions["career_priorities"] and priorities_list:
            priority_matches = []
            for p in priorities_list:
                p_lower = p.lower()
                if "helping" in p_lower or "social" in p_lower:
                    priority_matches.append(0.9 if ("Health" in career_family or "Social" in career_family or "Education" in career_family) else 0.4)
                elif "financial" in p_lower or "growth" in p_lower:
                    priority_matches.append(0.85 if ("Finance" in career_family or "Technology" in career_family or "Management" in career_family) else 0.5)
                elif "stability" in p_lower:
                    priority_matches.append(0.9 if ("Civil" in career_family or "Accounting" in career_family or "Health" in career_family) else 0.5)
                elif "creativity" in p_lower or "innovation" in p_lower:
                    priority_matches.append(0.9 if ("Design" in career_family or "Arts" in career_family or "Technology" in career_family) else 0.5)
                elif "practical" in p_lower:
                    priority_matches.append(0.85 if ("Engineering" in career_family or "Trades" in career_family) else 0.5)
                elif "research" in p_lower or "study" in p_lower:
                    priority_matches.append(0.85 if ("Science" in career_family or "Data" in career_family or "Research" in career_family) else 0.5)
            career_priority_score = sum(priority_matches) / max(len(priority_matches), 1) if priority_matches else 0.5
        else:
            career_priority_score = 0.5

        total_score = (
            active_weights.get("interests", 0.0) * interest_score
            + active_weights.get("academics", 0.0) * academic_score
            + active_weights.get("skills", 0.0) * skill_score
            + active_weights.get("extracurriculars", 0.0) * extracurricular_score
            + active_weights.get("work_preferences", 0.0) * work_preference_score
            + active_weights.get("career_priorities", 0.0) * career_priority_score
        )
        total_score = max(0.0, min(1.0, float(total_score)))

        why_reasons: list[str] = []
        if active_dimensions["interests"] and interest_score >= 0.6:
            matching_interests = [i for i in student_interests if i in curated_entry.get("relevant_interests", [])] if curated_entry else []
            if matching_interests:
                why_reasons.append(f"Strong overlap with your interest in {', '.join(matching_interests[:2])}.")
            else:
                why_reasons.append("High alignment with your selected interest dimensions.")
        if active_dimensions["academics"] and academic_score >= 0.6:
            why_reasons.append("Connects well with your academic strengths and subject inclinations.")
        if active_dimensions["skills"] and skill_score >= 0.6:
            matching_skills = [name for name, _ in top_skills if name.casefold() in {k.casefold() for k in student_skills}][:3]
            if matching_skills:
                why_reasons.append(f"Builds directly on your self-reported skills: {', '.join(matching_skills)}.")
            else:
                why_reasons.append("Strong fit with your core problem-solving competencies.")
        if active_dimensions["extracurriculars"] and extracurricular_score >= 0.6:
            why_reasons.append("Extracurricular activities provide valuable transferable skills for this occupation.")
        if active_dimensions["career_priorities"] and career_priority_score >= 0.6:
            why_reasons.append("Aligns with your long-term career priorities and work values.")
        if not why_reasons:
            why_reasons.append("Broad foundational alignment across your exploration profile.")

        areas_to_explore = [name for name, _ in top_skills if name.casefold() not in {k.casefold() for k in student_skills}][:4]

        results.append({
            "code": code,
            "title": title,
            "career_family": career_family,
            "description": str(occupation.get("Description", "")),
            "score": total_score,
            "dimension_scores": {
                "interests": round(interest_score, 3),
                "academics": round(academic_score, 3),
                "skills": round(skill_score, 3),
                "extracurriculars": round(extracurricular_score, 3),
                "work_preferences": round(work_preference_score, 3),
                "career_priorities": round(career_priority_score, 3),
            },
            "active_dimensions": active_dimensions,
            "weights_used": {k: round(v, 3) for k, v in active_weights.items()},
            "why_matched": why_reasons,
            "areas_to_explore": areas_to_explore,
            "top_skills": top_skills,
            "zone": zone_by_code.get(code, "—"),
            "zone_detail": zone_details.get(zone_by_code.get(code, ""), ""),
            "education": education_by_code.get(code, "Not reported in the included O*NET education table"),
            "software": software_by_code.get(code, []),
            "curated": curated_entry,
        })

    # Query Multi-Model AI Suite (CSE276 AI Foundations)
    suite = get_ai_model_suite()
    ml_preds_by_code: dict[str, dict[str, Any]] = {}
    ml_preds_by_title: dict[str, dict[str, Any]] = {}
    student_vec = None

    if suite.is_loaded:
        try:
            student_vec = suite.extract_features_from_student_profile(profile, curated)
            ml_preds = suite.predict_careers(student_vec, model_name=model_name, top_k=25)
            ml_preds_by_code = {p["onet_code"]: p for p in ml_preds if p.get("onet_code")}
            ml_preds_by_title = {p["title"]: p for p in ml_preds if p.get("title")}
        except Exception:
            student_vec = None

    # Instance-based O*NET KNN retrieval
    knn_scores = recommend_with_knn(data, profile, curated)

    for item in results:
        knn_score = knn_scores.get(item["code"])
        item["knn_similarity"] = round(float(knn_score), 4) if knn_score is not None else None
        
        ml_info = ml_preds_by_code.get(item["code"]) or ml_preds_by_title.get(item["title"])
        heuristic_score = float(item["score"])

        if ml_info:
            prob = float(ml_info["confidence_score"])
            item["ai_model_used"] = model_name
            item["ai_confidence_percent"] = ml_info["confidence_percent"]
            item["scoring_method"] = f"Hybrid AI Multi-Model Fusion ({model_name})"
            knn_part = float(knn_score) if knn_score is not None else 0.5
            # Fused score combining supervised probability, transparent rule-based score, and KNN similarity
            fused = 0.50 * prob + 0.30 * heuristic_score + 0.20 * knn_part
            item["score"] = max(0.0, min(1.0, float(fused)))
        else:
            item["ai_model_used"] = "O*NET KNN + Heuristic Baseline"
            if knn_score is not None:
                item["score"] = max(0.0, min(1.0, 0.70 * heuristic_score + 0.30 * float(knn_score)))
            else:
                item["score"] = max(0.0, min(1.0, heuristic_score))
            item["ai_confidence_percent"] = round(item["score"] * 100.0, 1)
            item["scoring_method"] = "Hybrid: weighted profile scoring + O*NET KNN"

        # Local Explainable AI (XAI) feature attributions
        if student_vec is not None and suite.is_loaded:
            try:
                xai = compute_local_explanation(student_vec, item["title"])
                item["explanation"] = xai
                item["drivers"] = xai.get("drivers", [])
                item["cautions"] = xai.get("cautions", [])
                item["category_contributions"] = xai.get("category_contributions", {})
            except Exception:
                item["explanation"] = {}
                item["drivers"] = []
                item["cautions"] = []
                item["category_contributions"] = {}
        else:
            item["explanation"] = {}
            item["drivers"] = []
            item["cautions"] = []
            item["category_contributions"] = {}

    return sorted(results, key=lambda item: (item["score"], item["title"]), reverse=True)[:max(1, int(limit))]
