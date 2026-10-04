# app.py
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Clinical Readmission Risk Engine",
    page_icon="🏥",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "readmission_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
LEADERBOARD_PATH = BASE_DIR / "data" / "processed" / "tournament_leaderboard.csv"


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error(
            "Model artifacts missing. Please run `python -m src.train_final` first."
        )
        st.stop()
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    return pipeline, meta


pipeline, meta = load_pipeline()
threshold = meta.get("optimal_threshold", 0.50)

# Sidebar: Governance & Audit Trail
st.sidebar.title("🏥 Clinical Model Card")
st.sidebar.markdown(f"**Architecture:** `{meta.get('model_type', 'XGBoost')}`")
st.sidebar.markdown(f"**Decision Threshold:** `{threshold:.3f}`")
st.sidebar.markdown(f"**Test ROC-AUC:** `{meta.get('test_roc_auc', 0.0):.3f}`")
st.sidebar.markdown(f"**Test PR-AUC:** `{meta.get('test_pr_auc', 0.0):.3f}`")
st.sidebar.markdown("---")

if LEADERBOARD_PATH.exists():
    st.sidebar.subheader("Tournament Leaderboard")
    lb_df = pd.read_csv(LEADERBOARD_PATH)
    st.sidebar.dataframe(lb_df, hide_index=True)

st.title("30-Day Hospital Readmission Risk Engine")
st.caption(
    "Clinical decision support tool for post-discharge intervention planning under the CMS Hospital Readmissions Reduction Program (HRRP)."
)

col_input, col_result = st.columns([1.6, 1.4], gap="large")

with col_input:
    st.subheader("Patient Encounter Profile")

    # Demographics & Stay
    c1, c2, c3 = st.columns(3)
    with c1:
        age_num = st.slider("Patient Age", min_value=18, max_value=95, value=65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox(
            "Race",
            ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
        )
    with c2:
        time_in_hospital = st.slider(
            "Length of Stay (Days)", min_value=1, max_value=14, value=4
        )
        admission_type_id = st.selectbox(
            "Admission Type",
            [1, 2, 3, 5],
            format_func=lambda x: {
                1: "Emergency",
                2: "Urgent",
                3: "Elective",
                5: "Trauma/Other",
            }.get(x, str(x)),
        )
        admission_source_id = st.selectbox(
            "Admission Source",
            [7, 1, 2],
            format_func=lambda x: {
                7: "Emergency Room",
                1: "Physician Referral",
                2: "Clinic Referral",
            }.get(x, str(x)),
        )
    with c3:
        num_diagnoses = st.slider(
            "Total Diagnoses Count", min_value=1, max_value=16, value=7
        )
        num_procedures = st.slider(
            "Procedures Count", min_value=0, max_value=6, value=1
        )
        num_medications = st.slider(
            "Medications Administered", min_value=1, max_value=80, value=14
        )

    st.markdown("---")
    st.markdown("#### Clinical Diagnoses & Lab Markers")
    d1, d2, d3 = st.columns(3)
    categories = [
        "Circulatory",
        "Respiratory",
        "Digestive",
        "Diabetes",
        "Injury",
        "Musculoskeletal",
        "Genitourinary",
        "Neoplasms",
        "Other",
    ]
    with d1:
        diag_1 = st.selectbox("Primary Diagnosis Chapter", categories, index=0)
        num_lab_procedures = st.slider(
            "Lab Procedures", min_value=1, max_value=130, value=45
        )
    with d2:
        diag_2 = st.selectbox(
            "Secondary Diagnosis Chapter", categories, index=3
        )
        A1Cresult = st.selectbox(
            "HbA1c Serum Result", ["None", "Norm", ">7", ">8"], index=0
        )
    with d3:
        diag_3 = st.selectbox("Tertiary Diagnosis Chapter", categories, index=1)
        max_glu_serum = st.selectbox(
            "Max Glucose Serum", ["None", "Norm", ">200", ">300"], index=0
        )

    st.markdown("---")
    st.markdown("#### Prior 12-Month Healthcare Utilization")
    u1, u2, u3 = st.columns(3)
    with u1:
        num_inpatient = st.number_input(
            "Prior Inpatient Admissions", min_value=0, max_value=20, value=1
        )
    with u2:
        num_emergency = st.number_input(
            "Prior ER Encounters", min_value=0, max_value=20, value=0
        )
    with u3:
        num_outpatient = st.number_input(
            "Prior Outpatient Visits", min_value=0, max_value=20, value=0
        )

    st.markdown("---")
    st.markdown("#### Pharmacotherapy & Inpatient Changes")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        insulin = st.selectbox("Insulin Regimen", ["No", "Steady", "Up", "Down"])
    with m2:
        metformin = st.selectbox(
            "Metformin", ["No", "Steady", "Up", "Down"], index=1
        )
    with m3:
        change = st.selectbox(
            "Medication Dosage Altered",
            ["No", "Ch"],
            format_func=lambda x: "Yes (Changed)" if x == "Ch" else "No",
        )
    with m4:
        diabetesMed = st.selectbox("Prescribed Diabetes Med", ["Yes", "No"])

# Construct payload for model inference
total_prior_visits = num_inpatient + num_emergency + num_outpatient

input_dict = {
    "age_num": [float(age_num)],
    "time_in_hospital": [int(time_in_hospital)],
    "num_lab_procedures": [int(num_lab_procedures)],
    "num_procedures": [int(num_procedures)],
    "num_medications": [int(num_medications)],
    "number_outpatient": [int(num_outpatient)],
    "number_emergency": [int(num_emergency)],
    "number_inpatient": [int(num_inpatient)],
    "number_diagnoses": [int(num_diagnoses)],
    "total_prior_visits": [int(total_prior_visits)],
    "race": [str(race)],
    "gender": [str(gender)],
    "admission_type_id": [str(admission_type_id)],
    "admission_source_id": [str(admission_source_id)],
    "max_glu_serum": [str(max_glu_serum)],
    "A1Cresult": [str(A1Cresult)],
    "metformin": [str(metformin)],
    "glipizide": ["No"],
    "glyburide": ["No"],
    "pioglitazone": ["No"],
    "rosiglitazone": ["No"],
    "insulin": [str(insulin)],
    "change": [str(change)],
    "diabetesMed": [str(diabetesMed)],
    "diag_1_group": [str(diag_1)],
    "diag_2_group": [str(diag_2)],
    "diag_3_group": [str(diag_3)],
}

input_df = pd.DataFrame(input_dict)

with col_result:
    st.subheader("Risk Stratification & Recommendation")

    # Run inference
    risk_prob = float(pipeline.predict_proba(input_df)[0, 1])
    is_high_risk = risk_prob >= threshold

    # Plotly Gauge Chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "30-Day Readmission Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": (
                        "#D9381E"
                        if is_high_risk
                        else (
                            "#F5A623"
                            if risk_prob >= threshold * 0.75
                            else "#2ECC71"
                        )
                    )
                },
                "steps": [
                    {"range": [0, threshold * 75], "color": "#E8F8F5"},
                    {
                        "range": [threshold * 75, threshold * 100],
                        "color": "#FEF9E7",
                    },
                    {"range": [threshold * 100, 100], "color": "#FDEDEC"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Risk Protocol Cards
    if is_high_risk:
        st.error(
            f"⚠️ **HIGH RISK OF 30-DAY READMISSION** (Score: {risk_prob*100:.1f}% vs Threshold: {threshold*100:.1f}%)"
        )
        st.markdown(
            """
        **Mandatory Clinical Action Plan:**
        - **Bedside Pharmacist Review:** Reconcile medications before discharge (especially insulin dosage modifications).
        - **Transition of Care Appointment:** Schedule in-person clinical follow-up within 72 hours.
        - **Care Coordinator Outbound:** Automated follow-up call at 48 hours post-discharge to verify medication adherence.
        """
        )
    elif risk_prob >= threshold * 0.75:
        st.warning(
            f"🟡 **MODERATE RISK (BORDERLINE)** (Score: {risk_prob*100:.1f}%)"
        )
        st.markdown(
            """
        **Recommended Clinical Protocol:**
        - Provide comprehensive diabetes self-management education.
        - Schedule virtual/telehealth appointment within 7–10 days.
        """
        )
    else:
        st.success(
            f"✅ **LOW READMISSION RISK** (Score: {risk_prob*100:.1f}%)"
        )
        st.markdown(
            """
        **Standard Discharge Protocol:**
        - Standard primary care follow-up within 30 days.
        - Standard patient education packet and prescription release.
        """
        )

    # Key Risk Drivers
    st.markdown("---")
    st.markdown("#### Patient Context Summary")
    st.write(
        f"- **Prior 12-month visits:** `{total_prior_visits}` visits (Inpatient: `{num_inpatient}`)"
    )
    st.write(f"- **Length of hospital stay:** `{time_in_hospital}` days")
    st.write(
        f"- **Medication change recorded:** `{'Yes' if change == 'Ch' else 'No'}`"
    )
    st.write(f"- **Primary organ chapter:** `{diag_1}`")