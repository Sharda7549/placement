import streamlit as st
import numpy as np
import pickle

# ---------------- Load the trained model ----------------
with open("placements.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- Page configuration ----------------
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

# ---------------- Inline HTML + CSS ----------------
st.markdown(
    """
    <style>
    /* -------- Global page background with animated blue gradient -------- */
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
        max-width: 500px;
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
st.markdown("<div class='card'><h1>🎓 Placement Predictor</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;font-size:1.1rem;'>"
    "Enter your <b>CGPA</b> and <b>IQ</b> to check your placement chances."
    "</p>",
    unsafe_allow_html=True
)

# ---------------- Inputs ----------------
col1, col2 = st.columns(2)
with col1:
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
with col2:
    iq = st.number_input("IQ", min_value=50, max_value=200, step=1)

# ---------------- Prediction ----------------
if st.button("Predict"):
    X = np.array([[cgpa, iq]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else 0.0
    pct = int(round(prob * 100))

    result_text = "✅ Will be Placed" if pred == 1 else "❌ Not Likely to be Placed"
    color = "#4dd0e1" if pred == 1 else "#ef5350"

    st.markdown(
        f"""
        <div class="popup" style="text-align:center;margin-top:1.2rem;">
            <h2 style="color:{color};margin-bottom:0.5rem;">{result_text}</h2>
        </div>
        <div class="meter-wrap">
            <div class="circle" style="--target:{pct}%;">
                <span>{pct}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)  # close card
