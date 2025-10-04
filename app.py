import streamlit as st
import numpy as np
import pickle

# ---------------- Load the trained model ----------------
with open("diabetes.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- Page configuration ----------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# ---------------- Inline HTML + CSS ----------------
st.markdown(
    """
    <style>
    /* -------- Global background with animated gradient -------- */
    body {
        background: linear-gradient(-45deg, #1e3c72, #2a5298, #4e73df, #1e3c72);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
        color: #fff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    @keyframes gradientBG {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }
    /* -------- Card container -------- */
    .card {
        background: rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 2rem;
        margin: auto;
        width: 90%;
        max-width: 650px;
        box-shadow: 0 0 20px rgba(0,0,0,0.25);
    }
    h1 {
        text-align: center;
        color: #fff;
        margin-bottom: 1rem;
    }
    label {
        color: #fff !important;
        font-weight: 600 !important;
    }
    /* Predict button */
    div.stButton > button:first-child {
        background-color: #0d47a1;
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1976d2;
        transform: scale(1.05);
    }
    /* -------- Circular Score Meter -------- */
    .meter-wrap {
        display:flex;
        justify-content:center;
        margin-top:1.5rem;
    }
    .circle {
        position: relative;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: conic-gradient(#4dd0e1 var(--val, 0%), rgba(255,255,255,0.15) 0%);
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fillMeter 2s forwards;
    }
    .circle span {
        position: absolute;
        font-size: 1.8rem;
        font-weight: bold;
        color: #fff;
    }
    @keyframes fillMeter {
        from { --val:0%; }
        to   { --val:var(--target); }
    }
    /* -------- Pop-up animation -------- */
    .popup {
        animation: pop 0.8s ease;
    }
    @keyframes pop {
        0% {transform: scale(0);}
        60%{transform: scale(1.1);}
        100%{transform: scale(1);}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Title & container ----------------
st.markdown("<div class='card'><h1>🩺 Diabetes Predictor</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;font-size:1.1rem;'>"
    "Enter patient details to predict the <b>likelihood of diabetes</b>."
    "</p>",
    unsafe_allow_html=True
)

# ---------------- Inputs ----------------
col1, col2 = st.columns(2)
with col1:
    glucose_bmi = st.number_input("Glucose × BMI Interaction", min_value=0.0, format="%.2f")
    glucose = st.number_input("Glucose", min_value=0.0, format="%.2f")
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    bmi_dpf = st.number_input("BMI × DiabetesPedigreeFunction", min_value=0.0, format="%.2f")

with col2:
    age = st.number_input("Age", min_value=0, step=1)
    preg_age = st.number_input("Pregnancies × Age Interaction", min_value=0.0, format="%.2f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    bp = st.number_input("Blood Pressure", min_value=0.0, format="%.2f")

# ---------------- Prediction ----------------
if st.button("Predict"):
    X = np.array([[glucose_bmi, glucose, bmi, bmi_dpf, age, preg_age, dpf, bp]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else 0.0
    pct = int(round(prob * 100))

    result_text = "⚠️ High Risk: Likely Diabetic" if pred == 1 else "✅ Low Risk: Unlikely Diabetic"
    color = "#ef5350" if pred == 1 else "#4dd0e1"

    st.markdown(
        f"""
        <div class="popup" style="text-align:center;margin-top:1.2rem;">
            <h2 style="color:{color};margin-bottom:0.5rem;">{result_text}</h2>
        </div>
        <div class="meter-wrap">
            <div class="circle" style="--target:{pct}%;background: conic-gradient({color} var(--target), rgba(255,255,255,0.15) 0%);">
                <span>{pct}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)  # close card
