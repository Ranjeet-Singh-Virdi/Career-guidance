"""Student assessment data models, educational stage definitions, and profile serialization."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EducationalStage(str, Enum):
    AFTER_CLASS_10 = "after_class_10"
    AFTER_CLASS_12 = "after_class_12"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"

    @classmethod
    def display_names(cls) -> dict[str, str]:
        return {
            cls.AFTER_CLASS_10.value: "After Class 10 (Secondary Education)",
            cls.AFTER_CLASS_12.value: "After Class 12 (Higher Secondary Education)",
            cls.UNDERGRADUATE.value: "Undergraduate student (College / University)",
            cls.GRADUATE.value: "Graduate or Postgraduate student (Advanced / Working)",
        }


INTEREST_CATEGORIES = [
    "Technology and computing",
    "Science and research",
    "Healthcare and biology",
    "Mathematics and analytical problem-solving",
    "Business and entrepreneurship",
    "Finance and accounting",
    "Arts and creative expression",
    "Design and architecture",
    "Communication, writing, and media",
    "Teaching and education",
    "Psychology and understanding people",
    "Social work and community development",
    "Law, government, and public service",
    "Sports and physical activities",
    "Agriculture and environmental sustainability",
    "Engineering and practical problem-solving",
    "Skilled trades and technical work",
]

STAGE_SUBJECTS = {
    EducationalStage.AFTER_CLASS_10.value: [
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Computer Science",
        "English and other languages",
        "History",
        "Geography",
        "Political Science",
        "Economics",
        "Fine Arts",
        "Physical Education",
        "Agriculture",
    ],
    EducationalStage.AFTER_CLASS_12.value: [
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Computer Science",
        "Accountancy",
        "Business Studies",
        "Economics",
        "History",
        "Political Science",
        "Geography",
        "Psychology",
        "Sociology",
        "Fine Arts",
        "Legal Studies",
        "English and other languages",
        "Physical Education",
        "Agriculture",
    ],
    EducationalStage.UNDERGRADUATE.value: [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Economics",
        "Accountancy",
        "Business Studies",
        "Psychology",
        "Sociology",
        "Political Science",
        "Legal Studies",
        "English and other languages",
        "Fine Arts",
        "Agriculture",
    ],
    EducationalStage.GRADUATE.value: [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Economics",
        "Accountancy",
        "Business Studies",
        "Psychology",
        "Sociology",
        "Political Science",
        "Legal Studies",
        "English and other languages",
    ],
}

EXTRACURRICULAR_ACTIVITIES = [
    "Sports and athletics",
    "Drawing, painting, and illustration",
    "Music and singing",
    "Dance and performing arts",
    "Coding and robotics",
    "Video editing and content creation",
    "Photography",
    "Writing, reading, and storytelling",
    "Public speaking and debate",
    "Volunteering and community service",
    "Leadership and student clubs",
    "Organizing events",
    "Entrepreneurship and selling products",
    "Building or repairing objects",
    "Gardening and environmental projects",
    "Research and science experiments",
    "Gaming and game development",
    "Craftwork and practical projects",
]

FOUNDATIONAL_SKILLS = [
    "Analytical thinking",
    "Creativity",
    "Communication",
    "Teamwork",
    "Leadership",
    "Problem-solving",
    "Numerical reasoning",
    "Writing",
    "Technical skills",
    "Organization",
    "Research",
    "Empathy",
    "Practical hands-on ability",
    "Attention to detail",
]


@dataclass
class SubjectPreference:
    name: str
    interest_level: int = 3       # 1 to 5
    confidence_level: int = 3     # 1 to 5
    academic_performance: float | None = None  # 0 to 100 percentage if provided


@dataclass
class ExtracurricularActivity:
    name: str
    enjoyment: int = 3            # 1 to 5
    experience: str = "intermediate"  # beginner, intermediate, advanced
    frequency: str = "weekly"     # occasional, weekly, daily
    wants_to_develop: bool = True


@dataclass
class WorkPreferences:
    team_vs_independent: str = "balanced"        # independent, teams, balanced, not_sure
    focus_dimension: str = "ideas"               # people, data, ideas, physical_objects, not_sure
    environment_type: str = "mixed"              # office, laboratory, outdoor, workshop, mixed, not_sure
    work_structure: str = "flexible"             # structured, creative, flexible, not_sure
    practical_vs_theory: str = "balanced"        # practical, theoretical, balanced, not_sure


@dataclass
class CareerPriorities:
    priorities: list[str] = field(default_factory=list)
    education_duration: str = "standard_degree"  # short_term, standard_degree, long_term_research, not_sure
    budget_preference: str = "moderate"          # affordable_scholarships, moderate, flexible, not_sure
    learning_mode: str = "classroom"             # classroom, online, hybrid, apprenticeship, not_sure


@dataclass
class StudentProfile:
    nickname: str = "Student"
    stage: str = EducationalStage.AFTER_CLASS_12.value
    academic_stream: str = ""
    preferred_region: str = "local_and_national"
    preferred_language: str = "English"
    interests: dict[str, int] = field(default_factory=dict)  # category -> rating (1-5)
    subjects: dict[str, SubjectPreference] = field(default_factory=dict)
    skills: dict[str, int] = field(default_factory=dict)     # skill name -> rating (1-5)
    extracurriculars: dict[str, ExtracurricularActivity] = field(default_factory=dict)
    work_preferences: WorkPreferences = field(default_factory=WorkPreferences)
    career_priorities: CareerPriorities = field(default_factory=CareerPriorities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nickname": self.nickname,
            "stage": self.stage,
            "academic_stream": self.academic_stream,
            "preferred_region": self.preferred_region,
            "preferred_language": self.preferred_language,
            "interests": self.interests,
            "subjects": {
                k: {
                    "interest": v.interest_level,
                    "confidence": v.confidence_level,
                    "performance": v.academic_performance,
                }
                for k, v in self.subjects.items()
            },
            "skills": self.skills,
            "extracurriculars": {
                k: {
                    "enjoyment": v.enjoyment,
                    "experience": v.experience,
                    "frequency": v.frequency,
                    "develop": v.wants_to_develop,
                }
                for k, v in self.extracurriculars.items()
            },
            "work_preferences": {
                "team_vs_independent": self.work_preferences.team_vs_independent,
                "focus_dimension": self.work_preferences.focus_dimension,
                "environment_type": self.work_preferences.environment_type,
                "work_structure": self.work_preferences.work_structure,
                "practical_vs_theory": self.work_preferences.practical_vs_theory,
            },
            "career_priorities": {
                "priorities": self.career_priorities.priorities,
                "education_duration": self.career_priorities.education_duration,
                "budget_preference": self.career_priorities.budget_preference,
                "learning_mode": self.career_priorities.learning_mode,
            },
        }


def parse_pipe_pairs(val: Any) -> dict[str, int]:
    """Parse string formatted as 'Name:Score|Name:Score'."""
    res: dict[str, int] = {}
    if not val or not isinstance(val, str):
        return res
    for item in val.split("|"):
        item = item.strip()
        if ":" in item:
            k, v = item.rsplit(":", 1)
            try:
                res[k.strip()] = int(float(v.strip()))
            except ValueError:
                continue
    return res


def parse_pipe_list(val: Any) -> list[str]:
    """Parse string formatted as 'Item1|Item2|Item3'."""
    if not val or not isinstance(val, str):
        return []
    return [p.strip() for p in val.split("|") if p.strip()]


def create_profile_from_demo(row: Mapping[str, Any]) -> StudentProfile:
    """Instantiate a StudentProfile from a CSV demo row."""
    nickname = str(row.get("display_name", "Student")).strip() or "Student"
    stage = str(row.get("stage", EducationalStage.AFTER_CLASS_12.value)).strip()
    if stage not in [s.value for s in EducationalStage]:
        stage = EducationalStage.AFTER_CLASS_12.value

    # Parse interests
    raw_interests = parse_pipe_list(row.get("interests", ""))
    interests: dict[str, int] = {}
    for name in raw_interests:
        interests[name] = 4  # default high rating for demo

    # Parse skills
    skills = parse_pipe_pairs(row.get("skill_ratings", ""))

    # Parse subjects
    raw_subjects = parse_pipe_pairs(row.get("subject_strengths", ""))
    subjects: dict[str, SubjectPreference] = {}
    for subj, perf in raw_subjects.items():
        subjects[subj] = SubjectPreference(
            name=subj,
            interest_level=4,
            confidence_level=4,
            academic_performance=float(perf),
        )

    # Parse extracurriculars
    raw_extra = parse_pipe_list(row.get("extracurriculars", ""))
    extracurriculars: dict[str, ExtracurricularActivity] = {}
    for act in raw_extra:
        extracurriculars[act] = ExtracurricularActivity(name=act, enjoyment=4, experience="intermediate")

    # Work preferences
    raw_work = parse_pipe_list(row.get("work_preferences", ""))
    wp = WorkPreferences()
    for w in raw_work:
        w_low = w.lower()
        if "data" in w_low or "ideas" in w_low:
            wp.focus_dimension = "ideas"
        elif "people" in w_low:
            wp.focus_dimension = "people"
        elif "object" in w_low or "physical" in w_low:
            wp.focus_dimension = "physical_objects"

        if "structured" in w_low:
            wp.work_structure = "structured"
        elif "creative" in w_low:
            wp.work_structure = "creative"

        if "office" in w_low:
            wp.environment_type = "office"
        elif "laboratory" in w_low or "lab" in w_low:
            wp.environment_type = "laboratory"
        elif "mixed" in w_low:
            wp.environment_type = "mixed"

        if "practical" in w_low:
            wp.practical_vs_theory = "practical"
        elif "theory" in w_low:
            wp.practical_vs_theory = "theoretical"

    # Career priorities
    raw_prio = parse_pipe_list(row.get("career_priorities", ""))
    cp = CareerPriorities(priorities=raw_prio)

    return StudentProfile(
        nickname=nickname,
        stage=stage,
        interests=interests,
        subjects=subjects,
        skills=skills,
        extracurriculars=extracurriculars,
        work_preferences=wp,
        career_priorities=cp,
    )
