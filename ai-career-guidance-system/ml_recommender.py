"""Unsupervised KNN career retrieval model built from O*NET occupation features.

No labelled student-outcome dataset is available, so this intentionally uses
nearest-neighbour retrieval rather than claiming supervised prediction accuracy.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

RIASEC = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]


def _numeric_rows(frame: pd.DataFrame, scales: set[str], prefix: str, nested_scale: str | None = None):
    """Return {occupation_code: {feature: normalized_value}} from an O*NET long table."""
    required = {"O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        return {}
    selected = frame[frame["Scale ID"].astype(str).str.upper().isin(scales)]
    values = pd.to_numeric(selected["Data Value"], errors="coerce")
    out: dict[str, dict[str, float]] = {}
    for code, name, value in zip(selected["O*NET-SOC Code"], selected["Element Name"], values):
        if pd.isna(value) or not str(code).strip() or not str(name).strip():
            continue
        # O*NET interest/importance ratings are generally on a 1-7 scale.
        out.setdefault(str(code), {})[f"{prefix}:{str(name).strip().casefold()}"] = float(value) / 7.0
    return out


def recommend_with_knn(data: Any, profile: Any, curated_data: dict | None = None) -> dict[str, float]:
    """Fit an occupation KNN index and return cosine-similarity scores by O*NET code."""
    curated = curated_data or {}
    interest_map = curated.get("interests", {})
    subject_map = curated.get("subjects", {})
    occupations = data.occupations
    if occupations is None or occupations.empty or "O*NET-SOC Code" not in occupations:
        return {}

    occ_features: dict[str, dict[str, float]] = {}
    sources = [
        _numeric_rows(data.interests, {"OI"}, "interest"),
        _numeric_rows(data.skills, {"IM"}, "skill"),
        _numeric_rows(data.knowledge, {"IM"}, "knowledge"),
    ]
    for source in sources:
        for code, feats in source.items():
            occ_features.setdefault(code, {}).update(feats)

    codes = [str(c) for c in occupations["O*NET-SOC Code"].tolist() if str(c) in occ_features]
    if not codes:
        return {}
    vocab = sorted({feature for code in codes for feature in occ_features[code]})
    col = {feature: i for i, feature in enumerate(vocab)}
    matrix = np.zeros((len(codes), len(vocab)), dtype=np.float32)
    for row, code in enumerate(codes):
        for feature, value in occ_features[code].items():
            matrix[row, col[feature]] = value
    matrix = normalize(matrix, norm="l2")

    p = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)
    query = np.zeros((1, len(vocab)), dtype=np.float32)
    def put(feature: str, value: float):
        idx = col.get(feature)
        if idx is not None:
            query[0, idx] = max(0.0, min(1.0, float(value)))

    for category, rating in p.get("interests", {}).items():
        info = interest_map.get(category, {})
        for riasec in info.get("onet_riasec", []):
            put(f"interest:{str(riasec).casefold()}", float(rating) / 5.0)
        if category in RIASEC:
            put(f"interest:{category.casefold()}", float(rating) / 5.0)

    for skill, rating in p.get("skills", {}).items():
        put(f"skill:{str(skill).casefold()}", float(rating) / 5.0)

    for subject, pref in p.get("subjects", {}).items():
        domains = subject_map.get(subject, {}).get("onet_knowledge", [])
        perf = pref.get("performance")
        try:
            academic = (0.4 * float(pref.get("interest", 3)) / 5.0
                        + 0.3 * float(pref.get("confidence", 3)) / 5.0
                        + 0.3 * float(perf) / 100.0) if perf not in (None, "") else (
                            0.5 * float(pref.get("interest", 3)) / 5.0
                            + 0.5 * float(pref.get("confidence", 3)) / 5.0)
        except (TypeError, ValueError):
            academic = 0.5
        for domain in domains:
            put(f"knowledge:{str(domain).casefold()}", academic)

    if not np.any(query):
        return {}
    query = normalize(query, norm="l2")
    # NearestNeighbors is fitted on O*NET occupation feature vectors.
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=len(codes))
    model.fit(matrix)
    distances, indices = model.kneighbors(query, n_neighbors=len(codes))
    return {codes[int(i)]: float(max(0.0, 1.0 - d)) for d, i in zip(distances[0], indices[0])}
