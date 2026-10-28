
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

from safety_rules import safety_check
from red_flags import detect_red_flags

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SmartTriage AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 SmartTriage AI")
st.subheader("AI-Powered OPD Triage Assistant")
st.warning("Hackathon demonstration only. Not for real clinical decisions.")

@st.cache_resource
def load_models():
    with open(BASE / "triage_level_model.pkl", "rb") as f:
        triage_model = pickle.load(f)
    with open(BASE / "department_model.pkl", "rb") as f:
        department_model = pickle.load(f)
    return triage_model, department_model

try:
    triage_model, department_model = load_models()
except Exception as e:
    st.error(f"Unable to load AI models: {e}")
    st.stop()

st.header("Patient Registration")

age = st.number_input("Patient Age (Years)", 0, 120, 25)
complaint = st.text_area("Chief Complaint", placeholder="Example: Severe chest pain")

col1, col2 = st.columns(2)

with col1:
    pulse = st.number_input("Pulse (BPM)", 0, 250, 80)
    spo2 = st.number_input("Oxygen Saturation (SpO2 %)", 0, 100, 98)

with col2:
    systolic = st.number_input("Systolic BP", 0, 260, 120)
    diastolic = st.number_input("Diastolic BP", 0, 180, 80)
    temperature = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0)

if st.button("Assess Patient", type="primary"):
    if not complaint.strip():
        st.error("Please enter the chief complaint.")
    else:
        try:
            patient = pd.DataFrame([{
    "age_years": age,
    "age_months": age * 12,
    "sex": "unknown",
    "pregnant": "unknown",
    "arrival_mode": "unknown",
    "chief_complaint": complaint,
    "medical_text": complaint,
    "pulse_bpm": pulse,
    "spo2_pct": spo2,
    "bp_systolic": systolic,
    "bp_diastolic": diastolic,
    "temp_c": temperature
}])

            predicted_level = int(triage_model.predict(patient)[0])
            predicted_department = str(department_model.predict(patient)[0])

            red_flags = detect_red_flags(complaint)

            safety = safety_check(
                age_years=age,
                spo2=spo2,
                pulse=pulse,
                systolic=systolic,
                diastolic=diastolic,
                temperature=temperature
            )

            final_level = min(
                predicted_level,
                int(safety["minimum_urgency_level"])
            )

            if red_flags:
                final_level = min(final_level, 2)

            if final_level <= 2:
                predicted_department = "Emergency/Casualty"

            st.divider()
            st.header("AI Triage Assessment")

            a, b = st.columns(2)
            a.metric("Triage Priority", f"Level {final_level}")
            b.metric("Recommended Department", predicted_department)

            if final_level <= 2:
                st.error("🚨 EMERGENCY PRIORITY — Immediate nurse review required.")
            elif final_level == 3:
                st.warning("Urgent — Nurse assessment recommended.")
            else:
                st.info("Lower priority — Await nurse confirmation.")

            if red_flags:
                st.error("Detected Red Flags: " + ", ".join(red_flags))
            else:
                st.write("Detected Red Flags: None")

            if safety["reasons"]:
                st.warning("Safety alerts: " + "; ".join(safety["reasons"]))

            if safety["needs_nurse_review"]:
                st.warning("Nurse review required.")

            st.caption("AI recommendations must be reviewed by trained staff.")

        except Exception as e:
            st.error(f"Assessment error: {e}")

