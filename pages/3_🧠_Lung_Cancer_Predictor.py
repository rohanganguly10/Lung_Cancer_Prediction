# -----------------------------
# 🧠 Lung Cancer Predictor Page
# -----------------------------
import streamlit as st
import numpy as np
import joblib
import requests
from streamlit_extras.colored_header import colored_header

# Optional GPIO for Raspberry Pi
try:
    import RPi.GPIO as GPIO
    IS_PI = True
except ImportError:
    IS_PI = False

# Load models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Streamlit page config
st.set_page_config(page_title="Lung Cancer Predictor", page_icon="🫁", layout="wide")

# Banner
colored_header(
    label="🫁 Lung Cancer Risk Predictor",
    description="AI-powered Smart Screening Tool with Raspberry Pi Integration",
    color_name="green-70",
)

# Sidebar model selection
st.sidebar.markdown("### ⚙️ Choose ML Model")
model_choice = st.sidebar.selectbox("Prediction Model:", ["Random Forest", "XGBoost"])

# Centered form container
form_container = st.columns([0.5, 5, 0.5])[1]

with st.form("prediction_form"):
    st.markdown("""
        <div style='background-color:#f0f4f7; padding:15px; border-radius:10px;'>
            <h4 style='color:#2E8B57;'>📋 Enter Patient Symptoms</h4>
        </div>
    """, unsafe_allow_html=True)

    patient_name = st.text_input("👤 Patient Name")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 20, 90, 50)
        smoking = st.radio("🚬 Smoking?", ["Yes", "No"])
        yellow_fingers = st.radio("🟡 Yellow Fingers?", ["Yes", "No"])
        anxiety = st.radio("😟 Anxiety?", ["Yes", "No"])

    with col2:
        peer_pressure = st.radio("🤝 Peer Pressure?", ["Yes", "No"])
        alcohol = st.radio("🍺 Alcohol?", ["Yes", "No"])
        chronic = st.radio("🏥 Chronic Disease?", ["Yes", "No"])
        fatigue = st.radio("💤 Fatigue?", ["Yes", "No"])
        allergy = st.radio("🤧 Allergy?", ["Yes", "No"])

    with col3:
        wheezing = st.radio("😤 Wheezing?", ["Yes", "No"])
        coughing = st.radio("🤒 Coughing?", ["Yes", "No"])
        breath = st.radio("😮‍💨 Shortness of Breath?", ["Yes", "No"])
        swallowing = st.radio("🥴 Swallowing Difficulty?", ["Yes", "No"])
        chest_pain = st.radio("💢 Chest Pain?", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Risk")

# Encode Yes/No
encode = lambda x: 1 if x == "Yes" else 0

# Prediction logic
if submitted and patient_name:
    features = [
        encode(yellow_fingers), encode(anxiety), encode(peer_pressure),
        encode(chronic), encode(fatigue), encode(allergy), encode(wheezing),
        encode(alcohol), encode(coughing), encode(swallowing), encode(chest_pain)
    ]
    derived = features[0] * features[1]
    final_input = np.array([features + [derived]])

    model = rf_model if model_choice == "Random Forest" else xgb_model
    pred = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][pred]

    st.markdown("---")
    st.subheader(f"🎯 Prediction Result for **{patient_name}**")

    if pred == 1:
        st.error(f"🚨 **High Risk** of Lung Cancer\n\n🧪 Confidence: `{prob*100:.2f}%`")
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdmh4bHR3bzBpZTIyZThwNW1lb2h0dmw3ZG0zN3l1bTJzbjBzZTd1ZCZlcD12MV8y/giphy.gif", width=280)

        if IS_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(18, GPIO.OUT)
            GPIO.output(18, GPIO.HIGH)

        # Hospital suggestions
        with st.expander("🏥 Find Nearby Cancer Hospitals"):
            city_input = st.text_input("📍 Enter your city or PIN code", value="Jaipur")
            if city_input:
                safe_city = city_input.replace(" ", "+")
                map_url = f"https://www.google.com/maps/search/cancer+hospitals+near+{safe_city}"
                st.markdown(f"🔗 [Hospitals near **{city_input}**]({map_url})", unsafe_allow_html=True)
                st.caption("ℹ️ Powered by Google Maps. Location based on your input.")

    else:
        st.success(f"✅ **Low Risk** of Lung Cancer\n\n🧪 Confidence: `{prob*100:.2f}%`")
        st.image("https://media.giphy.com/media/3ohzdYJK1wAdPWVk88/giphy.gif", width=280)

        if IS_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(23, GPIO.OUT)
            GPIO.output(23, GPIO.HIGH)

    st.caption(f"📌 Model Used: **{model_choice}**")

# GPIO cleanup
if IS_PI:
    import atexit
    atexit.register(GPIO.cleanup)