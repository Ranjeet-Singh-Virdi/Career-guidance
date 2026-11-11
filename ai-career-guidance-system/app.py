"""AI Career Guidance Platform.
Personalized career discovery, explainable recommendations and learning roadmaps.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from assessment import (
    EducationalStage,
    ExtracurricularActivity,
    STAGE_SUBJECTS,
    StudentProfile,
    SubjectPreference,
    WorkPreferences,
    CareerPriorities,
    create_profile_from_demo,
    INTEREST_CATEGORIES,
    EXTRACURRICULAR_ACTIVITIES,
    FOUNDATIONAL_SKILLS,
)
from career_engine import (
    DEFAULT_WEIGHTS,
    INTERESTS,
    SUBJECT_KNOWLEDGE,
    load_onet_data,
    recommend_student_profile,
)
from data_loader import (
    DATA_DIR,
    ONET_DIR,
    load_all_curated_data,
    load_demo_profiles,
)
from explainability import compute_local_explanation, FEATURE_DISPLAY_NAMES
from ml_models import get_ai_model_suite
from pathways_engine import (
    FREE_LEARNING_RESOURCES,
    calculate_skill_gap,
    generate_learning_roadmap,
    get_stage_pathways,
)
from train_models import train_and_benchmark_models

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="Career Compass | Personalized Career Guidance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End EdTech & AI Styling
st.markdown("""
<style>
:root {
    --primary: #0284c7;
    --primary-dark: #0369a1;
    --accent-teal: #0d9488;
    --accent-indigo: #6366f1;
    --accent-amber: #f59e0b;
    --dark-slate: #0f172a;
    --card-bg: #ffffff;
    --border-color: #e2e8f0;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}

/* Compact product-style hero */
.ai-hero {
    background: linear-gradient(115deg, #101d35 0%, #173c61 58%, #087e8b 100%);
    padding: 1.35rem 1.7rem;
    border-radius: 14px;
    color: #ffffff;
    margin-bottom: 1.1rem;
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.10);
}
.ai-hero h1 {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 750;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.025em;
}
.ai-hero p {
    color: #dbeafe;
    font-size: 0.94rem;
    margin: 0 0 0.65rem 0;
    max-width: 780px;
    line-height: 1.45;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.hero-badge {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #f8fafc;
}

/* Step Wizard Pill Badges */
.wizard-header {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.9rem 1.3rem;
    margin-bottom: 1.2rem;
}

/* Metric styling */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 14px;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

.badge-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    background: #e0f2fe;
    color: #0369a1;
    margin-right: 6px;
    margin-bottom: 6px;
}
.badge-teal {
    background: #ccfbf1;
    color: #0f766e;
}
.badge-amber {
    background: #fef3c7;
    color: #b45309;
}
.badge-indigo {
    background: #e0e7ff;
    color: #4338ca;
}

.info-banner {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 0.88rem;
    color: #14532d;
    margin: 1rem 0;
}
.disclaimer-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 0.88rem;
    color: #78350f;
    margin: 1rem 0;
}

/* Global density and typography refinements */
.block-container { padding-top: 0.85rem; padding-bottom: 2rem; max-width: 1240px; }
[data-testid="stMetric"] { padding: 10px 12px; border-radius: 10px; box-shadow: none; }
[data-testid="stMetricLabel"] { font-size: 0.78rem; }
[data-testid="stMetricValue"] { font-size: 1.45rem; }
[data-testid="stVerticalBlock"] { gap: 0.75rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }
[data-testid="stButton"] button { border-radius: 8px; font-weight: 600; }
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] { border-radius: 8px; }
[data-testid="stExpander"] { border-radius: 10px; }
.wizard-header { padding: 0.7rem 1rem; margin-bottom: 0.8rem; border-radius: 10px; }
.hero-badge { padding: 3px 9px; font-size: 0.72rem; }
@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .ai-hero { padding: 1.1rem; }
    .ai-hero h1 { font-size: 1.45rem; }
}
</style>
""", unsafe_allow_html=True)


# Data Loading Functions with Graceful Error Handling
@st.cache_data(show_spinner="Loading occupational database...")
def get_onet():
    return load_onet_data(ONET_DIR)


@st.cache_data(show_spinner=False)
def get_curated():
    return load_all_curated_data()


@st.cache_data(show_spinner=False)
def get_demos():
    return load_demo_profiles()


try:
    onet_data = get_onet()
    curated_data = get_curated()
    demo_profiles_df = get_demos()
except Exception as e:
    st.error(f"Critical error loading application data: {e}")
    st.info("Please verify that O*NET CSV files and data mappings exist in the `data/` directory.")
    st.stop()

# Load AI Model Suite
model_suite = get_ai_model_suite()

# Initialize Session State
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "profile" not in st.session_state:
    st.session_state.profile = StudentProfile()
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "🏠 Home"
if "custom_weights" not in st.session_state:
    st.session_state.custom_weights = dict(DEFAULT_WEIGHTS)

# Main navigation
with st.sidebar:
    st.markdown("### 🧭 Career Compass")
    st.caption("Explore options that match your interests and strengths")

    nav_options = [
        "🏠 Home",
        "🧭 Discover Careers",
        "🛠️ Learning Roadmap",
        "⚖️ Compare Careers",
        "📄 My Career Report",
    ]
    selected_nav = st.radio("Explore", nav_options, index=nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0)
    st.session_state.active_nav = selected_nav

    st.markdown("---")
    st.subheader("🧑‍🎓 Try a sample student")
    st.caption("Explore the portal with example student details")
    
    demo_names = ["-- Start Blank Profile --"]
    if not demo_profiles_df.empty and "display_name" in demo_profiles_df.columns:
        demo_names += demo_profiles_df["display_name"].tolist()

    chosen_demo = st.selectbox("Choose an example profile", demo_names, index=0)
    if st.button("Load Example Profile", use_container_width=True):
        if chosen_demo != "-- Start Blank Profile --":
            match_row = demo_profiles_df[demo_profiles_df["display_name"] == chosen_demo].iloc[0]
            st.session_state.profile = create_profile_from_demo(match_row)
            st.success(f"Loaded: {chosen_demo}")
            st.rerun()
        else:
            st.session_state.profile = StudentProfile()
            st.info("Reset to blank profile.")
            st.rerun()

    st.markdown("---")
    st.caption("🔒 Your answers are processed by this application. You can use the sample profile to explore before entering your own details.")


# Render Hero Banner on Top
st.markdown("""
<div class="ai-hero">
    <h1>🎓 AI Career Guidance</h1>
    <p>Discover career opportunities based on your interests, abilities, education and goals.</p>
    <div class="hero-badges">
        <span class="hero-badge">Made for students from every branch</span>
        <span class="hero-badge">Personalized career ideas</span>
        <span class="hero-badge">Simple learning plans</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Current Student Profile Reference
prof: StudentProfile = st.session_state.profile

# Compute recommendations using the fixed hybrid ensemble
recommendations = recommend_student_profile(
    onet_data,
    prof,
    curated_data=curated_data,
    custom_weights=st.session_state.custom_weights,
    limit=12,
    model_name="Hybrid AI Ensemble (Recommended)",
)

# Extract student feature vector for Explainability and telemetry
student_feature_vec = model_suite.extract_features_from_student_profile(prof, curated_data) if model_suite.is_loaded else None


# ==========================================
# PAGE 1: STUDENT ASSESSMENT WIZARD
# ==========================================
if st.session_state.active_nav == "🏠 Home":
    col_intro, col_cta = st.columns([3, 1])
    with col_intro:
        st.markdown(f"### Let’s get to know you, **{prof.nickname or 'Student'}**")
        st.write(
            "Complete a short guided assessment so we can understand what interests you, what you do well and what you want from your future."
        )
    st.info("For students from **all study backgrounds and engineering branches** — including CSE, AI/ML, ECE, Electrical, Mechanical, Civil, Science, Commerce and more. No programming or AI knowledge is required.")
    with col_cta:
        if st.button("🚀 Show My Career Matches", type="primary", use_container_width=True):
            st.session_state.active_nav = "🧭 Discover Careers"
            st.rerun()

    current_step = st.session_state.wizard_step
    step_titles = [
        "1. Educational Stage",
        "2. Interests & Things I Enjoy",
        "3. Academic Strengths",
        "4. Activities & Hobbies",
        "5. My Skills & Work Preferences",
        "6. What Matters in My Future",
    ]
    
    st.markdown(f"""
    <div class="wizard-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="color:#0f172a; font-size:1.05rem;">Assessment Step {current_step} of 6: {step_titles[current_step - 1]}</strong>
            <span style="color:#0284c7; font-weight:700;">{int((current_step / 6) * 100)}% Completed</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(current_step / 6.0)

    step_container = st.container(border=True)
    with step_container:
        # STEP 1: EDUCATIONAL STAGE
        if current_step == 1:
            st.markdown("### Step 1: Your Educational Stage & Background")
            c1, c2 = st.columns(2)
            with c1:
                nickname_in = st.text_input("Student Name or Identifier", value=prof.nickname)
                prof.nickname = nickname_in.strip() or "Student"
                
                stages_dict = EducationalStage.display_names()
                stage_keys = list(stages_dict.keys())
                current_idx = stage_keys.index(prof.stage) if prof.stage in stage_keys else 1
                selected_stage = st.selectbox("Current Educational Stage", stage_keys, index=current_idx, format_func=lambda k: stages_dict[k])
                prof.stage = selected_stage

                stream_in = st.text_input("Current Stream / Major (Optional)", value=prof.academic_stream, placeholder="e.g. Science PCM, Commerce with Math, CS Engineering...")
                prof.academic_stream = stream_in.strip()

            with c2:
                region_pref = st.selectbox(
                    "Preferred Study Destination",
                    ["India (National / Local)", "International (Abroad)", "Flexible / Both"],
                    index=0,
                )
                prof.preferred_region = region_pref

                lang_pref = st.selectbox(
                    "Preferred Instruction Language",
                    ["English", "Hindi / Regional Language", "Bilingual"],
                    index=0,
                )
                prof.preferred_language = lang_pref

        # STEP 2: CORE INTERESTS (RIASEC)
        elif current_step == 2:
            st.markdown("### Step 2: What subjects and activities interest you?")
            st.write("Rate how much you enjoy each area (1 = Not interested, 5 = Very interested). There are no right or wrong answers.")
            
            int_cols = st.columns(3)
            for idx, category in enumerate(INTEREST_CATEGORIES):
                with int_cols[idx % 3]:
                    current_val = prof.interests.get(category, 1)
                    val = st.slider(category, 1, 5, current_val, key=f"int_{category}")
                    prof.interests[category] = val

        # STEP 3: ACADEMIC STRENGTHS
        elif current_step == 3:
            st.markdown("### Step 3: Your Subjects & Study Experience")
            st.write("For each subject, share how interested and confident you feel. You can also add your marks if you know them.")
            
            stage_subjs = STAGE_SUBJECTS.get(prof.stage, STAGE_SUBJECTS[EducationalStage.AFTER_CLASS_12.value])
            sub_cols = st.columns(2)
            
            for idx, subj in enumerate(stage_subjs):
                with sub_cols[idx % 2]:
                    with st.expander(f"📚 {subj}", expanded=(subj in ["Mathematics", "Physics", "Computer Science", "Biology", "Accountancy"])):
                        pref = prof.subjects.get(subj, SubjectPreference(name=subj, interest_level=1, confidence_level=1))
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            int_lvl = st.slider("Interest", 1, 5, pref.interest_level, key=f"sub_int_{subj}")
                        with sc2:
                            conf_lvl = st.slider("Confidence", 1, 5, pref.confidence_level, key=f"sub_conf_{subj}")
                        with sc3:
                            perf_str = st.text_input("Marks % (Optional)", value=str(int(pref.academic_performance)) if pref.academic_performance else "", key=f"sub_perf_{subj}")
                        
                        perf_val = None
                        if perf_str.strip():
                            try:
                                perf_val = float(perf_str.strip())
                            except ValueError:
                                pass
                        prof.subjects[subj] = SubjectPreference(
                            name=subj,
                            interest_level=int_lvl,
                            confidence_level=conf_lvl,
                            academic_performance=perf_val,
                        )

        # STEP 4: EXTRACURRICULAR ACTIVITIES
        elif current_step == 4:
            st.markdown("### Step 4: Extracurricular Activities & Hobbies")
            st.write("Extracurriculars develop transferable competencies like leadership, creative synthesis, and problem solving:")
            
            selected_extras = st.multiselect(
                "Select activities you actively engage in or enjoy:",
                EXTRACURRICULAR_ACTIVITIES,
                default=list(prof.extracurriculars.keys()),
            )
            # Update extracurricular dict
            prof.extracurriculars = {
                act: prof.extracurriculars.get(act, ExtracurricularActivity(name=act, enjoyment=4))
                for act in selected_extras
            }

        # STEP 5: SKILLS & WORK STYLE
        elif current_step == 5:
            st.markdown("### Step 5: Skills You Use in Everyday Study and Life")
            st.write("Rate how confident you feel in each skill (1 = Still learning, 5 = Very confident). These skills can be developed over time.")
            
            sk_cols = st.columns(2)
            for idx, sk in enumerate(FOUNDATIONAL_SKILLS):
                with sk_cols[idx % 2]:
                    c_val = prof.skills.get(sk, 1)
                    val = st.slider(sk, 1, 5, c_val, key=f"sk_{sk}")
                    prof.skills[sk] = val

            st.markdown("---")
            st.markdown("#### What kind of work setting sounds good to you?")
            w1, w2 = st.columns(2)
            with w1:
                focus_choice = st.selectbox(
                    "What do you enjoy working on most?",
                    ["ideas (concepts, strategy, algorithms)", "data (analytics, code, finance)", "people (mentoring, healthcare, law)", "physical_objects (engineering, hardware, labs)"],
                    index=0,
                )
                prof.work_preferences.focus_dimension = focus_choice.split()[0]
            with w2:
                env_choice = st.selectbox(
                    "Where would you prefer to work?",
                    ["office (tech / corporate / studio)", "laboratory (scientific research / clinical)", "workshop (makerspace / fabrication)", "mixed"],
                    index=0,
                )
                prof.work_preferences.environment_type = env_choice.split()[0]

        # STEP 6: CAREER PRIORITIES
        elif current_step == 6:
            st.markdown("### Step 6: What Is Important for Your Future?")
            priority_options = [
                "innovation (building cutting-edge technology and AI systems)",
                "financial_growth (high compensation and rapid wealth creation)",
                "helping_people (healthcare, clinical psychology, public service)",
                "job_stability (predictable governance, tenure, established firms)",
                "social_impact (civic reform, sustainability, public policy)",
                "creativity (design, original media, architecture)",
                "further_study (academic research and postgraduate specialization)",
            ]
            existing_p = [p for p in priority_options if p.split()[0] in prof.career_priorities.priorities] or priority_options[:2]
            chosen_p = st.multiselect("Choose 2–4 things that matter most to you when thinking about a future career:", priority_options, default=existing_p)
            prof.career_priorities.priorities = [p.split()[0] for p in chosen_p]

            st.success("🎉 Assessment profile completed! Click below to compute your personalized recommendations.")

    # Wizard Step Buttons (Previous / Next)
    b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
    with b_col1:
        if current_step > 1:
            if st.button("⬅️ Previous Step", use_container_width=True):
                st.session_state.wizard_step -= 1
                st.rerun()
    with b_col3:
        if current_step < 6:
            if st.button("Next Step ➡️", type="primary", use_container_width=True):
                st.session_state.wizard_step += 1
                st.rerun()
        else:
            if st.button("✨ Find Careers That Match Me", type="primary", use_container_width=True):
                st.session_state.active_nav = "🧭 Discover Careers"
                st.rerun()


# ==========================================
# PAGE 2: AI CAREER RECOMMENDATIONS
# ==========================================
elif st.session_state.active_nav == "🧭 Discover Careers":
    st.subheader(f"🧭 Career Matches for {prof.nickname}")
    st.caption("Personalized suggestions based on your interests, strengths, preferences and career goals.")

    if not recommendations:
        st.warning("Please complete the student assessment wizard or apply a demo profile to generate career recommendations.")
        st.stop()

    # Career overview
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Top Recommended Career", recommendations[0]["title"], f"{recommendations[0]['score']:.1%}")
    with m2:
        st.metric("Profile Alignment", f"{recommendations[0]['score']:.1%}")
    with m3:
        lead_family = recommendations[0].get("career_family", "General")
        st.metric("Lead Career Family", lead_family)
    with m4:
        st.metric("Career Library", f"{len(onet_data.occupations):,} roles")

    st.markdown("---")
    st.markdown("### 🌟 Careers that align with your profile")

    top3_cols = st.columns(3)
    for idx, rec in enumerate(recommendations[:3]):
        with top3_cols[idx]:
            with st.container(border=True):
                st.markdown(f"#### #{idx+1} {rec['title']}")
                st.caption(f"**Career area:** {rec.get('career_family', 'General')}")
                
                # Confidence Progress
                st.write(f"**Career Match Score:** {rec['score']:.1%}")
                st.progress(float(rec["score"]))
                
                st.write(rec["description"][:160] + "..." if len(rec["description"]) > 160 else rec["description"])

                # Top Positive Drivers (XAI)
                if rec.get("drivers"):
                    st.markdown("**Key Recommendation Drivers:**")
                    for d in rec["drivers"][:2]:
                        st.markdown(f"• <span style='color:#0d9488; font-weight:600;'>{d['display_name']} ({d['formatted_value']})</span>", unsafe_allow_html=True)


                if rec.get("education"):
                    st.caption(f"**Path:** {rec['education']}")

    st.markdown("---")
    st.markdown("### 📋 More Careers You Can Explore")
    
    # Filter Controls
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        all_families = sorted(list(set(r.get("career_family", "General") for r in recommendations)))
        fam_filter = st.multiselect("Show career areas", all_families, default=all_families)
    with f_col2:
        min_score = st.slider("Minimum match score", 0, 100, 30, 5) / 100.0

    filtered_recs = [r for r in recommendations if r.get("career_family", "General") in fam_filter and r["score"] >= min_score]

    for rank, rec in enumerate(filtered_recs, 1):
        with st.expander(f"#{rank} {rec['title']} — {rec['score']:.1%} Match ({rec.get('career_family', 'General')})", expanded=(rank == 1)):
            c_left, c_right = st.columns([3, 2])
            with c_left:
                st.markdown(f"**Description:** {rec['description']}")
                st.markdown("**Why this career appeared:**")
                for reason in rec.get("why_matched", []):
                    st.write(f"• {reason}")

                if rec.get("areas_to_explore"):
                    st.markdown("**Skills to develop:**")
                    st.write(", ".join(rec["areas_to_explore"]))

            with c_right:
                # Dimension radar / bar chart for this career
                dim_scores = rec.get("dimension_scores", {})
                if dim_scores:
                    fig_dim = go.Figure(data=go.Scatterpolar(
                        r=list(dim_scores.values()),
                        theta=[k.capitalize().replace("_", " ") for k in dim_scores.keys()],
                        fill='toself',
                        fillcolor='rgba(2, 132, 199, 0.25)',
                        line=dict(color='#0284c7', width=2),
                    ))
                    fig_dim.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                        height=260,
                        margin=dict(l=30, r=30, t=25, b=25),
                    )
                    st.plotly_chart(fig_dim, use_container_width=True)


# ==========================================
# PAGE 3: EXPLAINABLE AI (XAI) RATIONALE
# ==========================================
# ==========================================
# PAGE 4: SKILL-GAP & ROADMAPS
# ==========================================
elif st.session_state.active_nav == "🛠️ Learning Roadmap":
    st.subheader("🛠️ Your Personalized Learning Roadmap")
    st.caption("See which skills are useful for a career and follow a simple four-week plan to start building them.")

    if not recommendations:
        st.warning("Please complete the assessment or load a demo profile.")
        st.stop()

    career_titles = [r["title"] for r in recommendations]
    chosen_career = st.selectbox("Which career would you like to prepare for?", career_titles, index=0)
    target_rec = next(r for r in recommendations if r["title"] == chosen_career)

    skill_gaps = calculate_skill_gap(target_rec, prof.skills)
    roadmap = generate_learning_roadmap(target_rec, skill_gaps)

    # Radar Comparison: Student Skills vs O*NET Target
    st.markdown(f"### Skills to Explore for: **{target_rec['title']}**")
    
    if skill_gaps:
        top_gaps = skill_gaps[:8]
        labels = [g["skill_name"] for g in top_gaps]
        student_vals = [g["student_rating"] for g in top_gaps]
        target_vals = [g["target_level"] for g in top_gaps]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=student_vals,
            theta=labels,
            fill='toself',
            name='Your Current Level',
            line=dict(color='#0284c7'),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=target_vals,
            theta=labels,
            fill='toself',
            name='O*NET Industry Requirement',
            line=dict(color='#ef4444', dash='dash'),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="Skill Proficiency Benchmark vs O*NET Occupational Standard",
            height=380,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Skill Gap Action Cards
    st.markdown("### 🎯 Skills You Can Practise")
    gap_cols = st.columns(3)
    for idx, gap in enumerate(skill_gaps[:6]):
        with gap_cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"**{gap['skill_name']}**")
                st.caption(f"Status: **{gap['status']}**")
                st.write(f"Your Rating: **{gap['student_rating']}/5** | Required: **{gap['target_level']}/5**")
                st.progress(min(1.0, gap["student_rating"] / 5.0))
                st.caption(f"💡 {gap['action_tip']}")

    st.markdown("---")
    st.markdown("### 📅 Your 4-Week Starter Plan")
    for step in roadmap["weekly_schedule"]:
        with st.container(border=True):
            st.markdown(f"#### {step['week']}: {step['title']}")
            st.markdown(step["task"])

    st.markdown("---")
    st.markdown("### 📚 Free Websites and Courses to Learn More")
    res_cols = st.columns(2)
    for idx, res in enumerate(FREE_LEARNING_RESOURCES):
        with res_cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**[{res['name']}]({res['url']})**")
                st.caption(f"Domain: {res['category']}")
                st.write(res["description"])


# ==========================================
# PAGE 5: AI & ML MODELS HUB
# ==========================================

# ==========================================
# PAGE 6: CAREER COMPARISON MATRIX
# ==========================================
elif st.session_state.active_nav == "⚖️ Compare Careers":
    st.subheader("⚖️ Compare Your Career Options")
    st.caption("Compare career options side by side, including their match scores, education paths and useful skills.")

    if not recommendations:
        st.warning("Please complete the assessment or load a demo profile.")
        st.stop()

    all_career_names = [r["title"] for r in recommendations]
    selected_for_comp = st.multiselect(
        "Choose 2–4 careers to compare:",
        all_career_names,
        default=all_career_names[:3] if len(all_career_names) >= 3 else all_career_names,
    )

    if len(selected_for_comp) >= 2:
        comp_recs = [r for r in recommendations if r["title"] in selected_for_comp]
        
        # Radar Chart Comparison
        fig_multi = go.Figure()
        colors = ['#0284c7', '#0d9488', '#f59e0b', '#6366f1']
        for i, rec in enumerate(comp_recs):
            dim_s = rec.get("dimension_scores", {})
            fig_multi.add_trace(go.Scatterpolar(
                r=list(dim_s.values()),
                theta=[k.capitalize().replace("_", " ") for k in dim_s.keys()],
                fill='toself',
                name=rec["title"],
                line=dict(color=colors[i % len(colors)]),
            ))
        fig_multi.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Multidimensional Profile Alignment Comparison",
            height=440,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_multi, use_container_width=True)

        # Attribute Table
        comp_rows = []
        for rec in comp_recs:
            comp_rows.append({
                "Career Title": rec["title"],
                "Career Family": rec.get("career_family", "General"),
                "Career Match Score": f"{rec['score']:.1%}",
                "O*NET Zone": f"Zone {rec['zone']}",
                "Prep Pathway": rec.get("education", "Standard University Degree"),
                "Core Skills": ", ".join([s[0] for s in rec.get("top_skills", [])[:3]]),
            })
        st.table(pd.DataFrame(comp_rows))


# ==========================================
# PAGE 7: SYSTEM ARCHITECTURE & REPORT
# ==========================================
elif st.session_state.active_nav == "📄 My Career Report":
    st.subheader("📄 Your Career Guidance Report")
    
    st.caption("A personal snapshot of your career exploration, recommended directions and suggested preparation steps.")
    if not recommendations:
        st.info("Complete your career assessment to create a personalized report.")

    st.markdown("---")
    st.markdown("### 📥 Save your guidance report")

    # CSV Download
    csv_rows = []
    for r in recommendations:
        csv_rows.append({
            "Rank": len(csv_rows) + 1,
            "Title": r["title"],
            "Family": r.get("career_family", "General"),
            "Match_Score": f"{r['score']:.1%}",
            "Education": r.get("education", ""),
        })
    csv_data = pd.DataFrame(csv_rows).to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Career Matches (CSV)",
        data=csv_data,
        file_name=f"career_guidance_report_{prof.nickname.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # Detailed Markdown Report
    report_text = f"""# Career Compass - My Career Guidance Report
**Student Identifier:** {prof.nickname}
**Educational Stage:** {EducationalStage.display_names().get(prof.stage, prof.stage)}
**Guidance approach:** Personalized multi-signal career matching
**Generated Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Top Recommended Career Pathways
"""
    for i, r in enumerate(recommendations[:5], 1):
        report_text += f"\n### #{i} {r['title']} ({r.get('career_family', 'General')})\n"
        report_text += f"- **Career Match Score:** {r['score']:.1%}\n"

        report_text += f"- **Description:** {r['description']}\n"
        report_text += f"- **Preparation:** {r.get('education', 'Standard academic degree')}\n"
        if r.get('drivers'):
            report_text += f"- **Top Drivers:** {', '.join([d['display_name'] for d in r['drivers'][:3]])}\n"

    st.download_button(
        "📄 Download Full Career Report",
        data=report_text.encode('utf-8'),
        file_name=f"career_dossier_{prof.nickname.lower().replace(' ', '_')}.md",
        mime="text/markdown",
    )
