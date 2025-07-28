import streamlit as st
import numpy as np
import joblib

# Optional GPIO support for Raspberry Pi
try:
    import RPi.GPIO as GPIO
    IS_PI = True
except ImportError:
    IS_PI = False

# Load trained models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Streamlit page config
st.set_page_config(
    page_title="Lung Cancer Predictor",
    page_icon="🫁",
    layout="centered"
)

# Title and description
st.markdown("<h1 style='text-align:center; color:#2E8B57;'>🫁 Lung Cancer Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Smart ML-powered prediction system • Raspberry Pi-ready</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for model selection
st.sidebar.header("🧠 Select Model")
model_choice = st.sidebar.radio("Choose the prediction model:", ["Random Forest", "XGBoost"])

# Input form
with st.form("lung_form"):
    st.subheader("📋 Patient Input Form")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", min_value=20, max_value=90, value=50)
        smoking = st.radio("Do you smoke?", ["Yes", "No"])
        yellow_fingers = st.radio("Do you have yellow fingers?", ["Yes", "No"])
        anxiety = st.radio("Do you experience anxiety?", ["Yes", "No"])
        peer_pressure = st.radio("Do you face peer pressure?", ["Yes", "No"])
        alcohol = st.radio("Do you consume alcohol?", ["Yes", "No"])

    with col2:
        chronic = st.radio("Chronic disease?", ["Yes", "No"])
        fatigue = st.radio("Fatigue?", ["Yes", "No"])
        allergy = st.radio("Allergy?", ["Yes", "No"])
        wheezing = st.radio("Wheezing?", ["Yes", "No"])
        coughing = st.radio("Frequent coughing?", ["Yes", "No"])
        breath = st.radio("Shortness of breath?", ["Yes", "No"])
        swallowing = st.radio("Swallowing difficulty?", ["Yes", "No"])
        chest_pain = st.radio("Chest pain?", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Risk")

# Helper to convert Yes/No to 1/0
def bin_val(val): return 1 if val == "Yes" else 0

# Prediction logic
if submitted:

    # Collect only the final 11 actual + 1 derived features
    input_data = [
    bin_val(yellow_fingers),
    bin_val(anxiety),
    bin_val(peer_pressure),
    bin_val(chronic),
    bin_val(fatigue),
    bin_val(allergy),
    bin_val(wheezing),
    bin_val(alcohol),
    bin_val(coughing),
    bin_val(swallowing),
    bin_val(chest_pain),
    ]

    # Derived feature
    anxyelfin = input_data[0] * input_data[1]
    full_input = np.array([input_data + [anxyelfin]])


    # Choose model
    model = rf_model if model_choice == "Random Forest" else xgb_model
    pred = model.predict(full_input)[0]
    prob = model.predict_proba(full_input)[0][pred]

    # Result display
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if pred == 1:
        st.error(f"🚨 **High Risk** of Lung Cancer\n🧮 Confidence: `{prob*100:.2f}%`")
        if IS_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(18, GPIO.OUT)
            GPIO.output(18, GPIO.HIGH)  # Turn on warning light/buzzer
    else:
        st.success(f"✅ **Low Risk** of Lung Cancer\n🧮 Confidence: `{prob*100:.2f}%`")
        if IS_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(23, GPIO.OUT)
            GPIO.output(23, GPIO.HIGH)  # Turn on green LED

    st.caption(f"🧪 Model used: **{model_choice}**")

# Cleanup GPIO (optional, run once at end)
if IS_PI:
    import atexit
    atexit.register(GPIO.cleanup)

