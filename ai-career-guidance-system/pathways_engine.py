"""Stage-specific educational pathways and skill-gap growth planning logic."""
from __future__ import annotations

from typing import Any, Mapping

from assessment import EducationalStage, StudentProfile
from data_loader import load_all_curated_data

FREE_LEARNING_RESOURCES = [
    {
        "category": "Foundational & School Curriculum",
        "name": "NCERT e-Pathshala & DIKSHA",
        "description": "Free official textbooks, audiobooks, and interactive teaching material for Classes 1 to 12 across all streams.",
        "url": "https://epathshala.nic.in",
    },
    {
        "category": "Higher Education & Engineering",
        "name": "NPTEL & SWAYAM (Govt. of India / IITs)",
        "description": "Free university-level video courses and lecture series produced by faculty from IITs, IISc, and central universities.",
        "url": "https://nptel.ac.in",
    },
    {
        "category": "Mathematics & Core Sciences",
        "name": "Khan Academy",
        "description": "Mastery-based practice exercises, instructional videos, and personalized learning dashboard in math, science, and computing.",
        "url": "https://www.khanacademy.org",
    },
    {
        "category": "Computer Science & Programming",
        "name": "freeCodeCamp",
        "description": "Interactive coding curriculum with verified project-based certifications in Python, JavaScript, and Machine Learning.",
        "url": "https://www.freecodecamp.org",
    },
    {
        "category": "Advanced University Lectures",
        "name": "MIT OpenCourseWare",
        "description": "Free, open publication of educational material from thousands of MIT undergraduate and graduate courses.",
        "url": "https://ocw.mit.edu",
    },
    {
        "category": "Career Exploration & Schemes",
        "name": "National Career Service (NCS India)",
        "description": "Official Ministry of Labour & Employment portal offering career counseling, apprenticeship notifications, and sector insights.",
        "url": "https://www.ncs.gov.in",
    },
]

DOMAIN_PROJECT_TEMPLATES = {
    "programming": "Build a basic calculator or simple interactive personal website in Python/JavaScript",
    "design": "Create a digital poster, brand identity, or UI design mockup in a free tool like Figma or Canva",
    "communication": "Prepare a 5-minute presentation or draft an argumentative essay for a community topic",
    "science": "Conduct a safe, age-appropriate science experiment or environmental observation log",
    "business": "Develop a simple one-page business plan or personal budget tracker spreadsheet",
    "environment": "Document a local waste or energy challenge and research three feasible community solutions",
}


def get_stage_pathways(
    stage: str,
    top_recommendations: list[dict[str, Any]],
    curated_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve and score educational pathways tailored to the student's current stage and top careers."""
    curated = curated_data or load_all_curated_data()
    all_pathways = curated.get("pathways", {})
    stage_key = stage.lower()

    if stage_key in [EducationalStage.AFTER_CLASS_10.value, "class_10", "10th"]:
        stage_items = all_pathways.get("class_10", [])
    elif stage_key in [EducationalStage.AFTER_CLASS_12.value, "class_12", "12th"]:
        stage_items = all_pathways.get("class_12", [])
    else:
        stage_items = all_pathways.get("graduate", [])

    if not stage_items:
        return []

    recommended_families = set()
    for rec in top_recommendations[:5]:
        fam = rec.get("career_family")
        if fam:
            recommended_families.add(fam.casefold())

    scored_pathways = []
    for p in stage_items:
        p_fams = [f.casefold() for f in p.get("career_families", [])]
        overlap = sum(1 for f in p_fams if f in recommended_families)
        match_score = min(1.0, 0.4 + (overlap * 0.25))

        scored_pathways.append({
            **p,
            "relevance_score": match_score,
            "matching_families": [f for f in p.get("career_families", []) if f.casefold() in recommended_families],
        })

    return sorted(scored_pathways, key=lambda x: x["relevance_score"], reverse=True)


def calculate_skill_gap(
    career: dict[str, Any],
    student_skills: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Compare student self-ratings against career skill demands constructively without negative labeling."""
    top_skills = career.get("top_skills", []) or career.get("skills", [])
    if not top_skills:
        return []

    norm_student_skills = {k.casefold(): int(v) for k, v in student_skills.items()}
    gaps = []

    for item in top_skills:
        if isinstance(item, (tuple, list)):
            skill_name = str(item[0])
            scales = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
        else:
            skill_name = str(item)
            scales = {}

        req_raw = scales.get("LV", scales.get("IM", 4.0))
        target_level = round(float(req_raw) * 5.0 / 7.0, 1)

        current_rating = norm_student_skills.get(skill_name.casefold(), 3)
        gap_diff = target_level - current_rating

        if gap_diff > 1.2:
            status = "Growth Opportunity"
            badge_color = "orange"
            action_tip = "A high-leverage skill to explore through structured online tutorials and practical mini-projects."
        elif gap_diff > 0.3:
            status = "Skill in Progress"
            badge_color = "blue"
            action_tip = "Good emerging foundation; continue deepening your hands-on practice."
        else:
            status = "Strong Alignment"
            badge_color = "green"
            action_tip = "A well-developed strength you can leverage and showcase in portfolios."

        gaps.append({
            "skill_name": skill_name,
            "student_rating": current_rating,
            "target_level": target_level,
            "gap_difference": round(gap_diff, 1),
            "status": status,
            "badge_color": badge_color,
            "action_tip": action_tip,
        })

    return sorted(gaps, key=lambda x: x["gap_difference"], reverse=True)


def generate_learning_roadmap(
    career: dict[str, Any],
    skill_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a structured, encouraging 4-week short-term action plan for student growth."""
    curated = career.get("curated") or {}
    beginner_projects = list(curated.get("suggested_beginner_projects", []))

    if not beginner_projects:
        title_lower = career.get("title", "").lower()
        fam_lower = career.get("career_family", "").lower()
        if "software" in title_lower or "program" in title_lower or "comput" in title_lower or "data" in title_lower:
            beginner_projects.append(DOMAIN_PROJECT_TEMPLATES["programming"])
        elif "design" in title_lower or "art" in title_lower:
            beginner_projects.append(DOMAIN_PROJECT_TEMPLATES["design"])
        elif "writ" in title_lower or "communicat" in title_lower or "media" in title_lower or "law" in title_lower:
            beginner_projects.append(DOMAIN_PROJECT_TEMPLATES["communication"])
        elif "business" in fam_lower or "finance" in fam_lower or "account" in title_lower:
            beginner_projects.append(DOMAIN_PROJECT_TEMPLATES["business"])
        elif "environment" in fam_lower or "agricultur" in fam_lower:
            beginner_projects.append(DOMAIN_PROJECT_TEMPLATES["environment"])
        else:
            beginner_projects.append(DOMAIN_PROJECT_TEMPLATES["science"])

    primary_project = beginner_projects[0]
    top_growth_skill = skill_gaps[0]["skill_name"] if skill_gaps else "Problem Solving"

    week_1 = f"**Explore & Foundation:** Dedicate 3–4 hours to free introductory lessons on **{top_growth_skill}** using Khan Academy or NPTEL. Note down 3 key concepts you learned."
    week_2 = f"**Guided Practice:** Complete a hands-on exercise or solve small challenges applying **{top_growth_skill}**. Review solutions and identify areas to refine."
    week_3 = f"**Mini-Project:** Begin work on your beginner project: *\"{primary_project}\"*. Document your step-by-step thinking in a small project journal."
    week_4 = f"**Review & Iterate:** Share your project draft with a teacher, mentor, or peer for constructive feedback. Reflect on your progress and update your self-confidence rating."

    return {
        "focus_skill": top_growth_skill,
        "primary_project": primary_project,
        "all_projects": beginner_projects,
        "weekly_schedule": [
            {"week": "Week 1", "title": "Explore & Foundation", "task": week_1},
            {"week": "Week 2", "title": "Guided Practice", "task": week_2},
            {"week": "Week 3", "title": "Beginner Project Execution", "task": week_3},
            {"week": "Week 4", "title": "Review, Feedback & Next Steps", "task": week_4},
        ],
        "curated_resources": FREE_LEARNING_RESOURCES,
    }
