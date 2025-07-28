import streamlit as st
import numpy as np
import joblib
import requests
from streamlit_extras.colored_header import colored_header

# Optional GPIO
try:
    import RPi.GPIO as GPIO
    IS_PI = True
except ImportError:
    IS_PI = False

# Load models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Page setup
st.set_page_config(page_title="Lung Cancer Predictor", page_icon="🫁", layout="wide")

colored_header(
    label="🫁 Lung Cancer Risk Predictor",
    description="AI-powered Smart Screening Tool with Raspberry Pi Integration",
    color_name="green-70",
)

# Sidebar model selection
st.sidebar.markdown("### ⚙️ Choose ML Model")
model_choice = st.sidebar.selectbox("Prediction Model:", ["Random Forest", "XGBoost"])

form_container = st.columns([0.5, 5, 0.5])[1]  # Narrow and centered form

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

# Encode helper
encode = lambda x: 1 if x == "Yes" else 0

# 🔍 Predict
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
        if IS_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(18, GPIO.OUT)
            GPIO.output(18, GPIO.HIGH)
            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdmh4bHR3bzBpZTIyZThwNW1lb2h0dmw3ZG0zN3l1bTJzbjBzZTd1ZCZlcD12MV8y/giphy.gif", width=280)

        # 📍 Suggest hospitals nearby
        try:
            location_data = requests.get("https://ipinfo.io").json()
            city = location_data.get("city", "")
            region = location_data.get("region", "")
            map_url = f"https://www.google.com/maps/search/cancer+hospitals+near+{city}+{region}"
            st.markdown("### 🏥 Nearby Cancer Care")
            st.markdown(f"🔎 [Find cancer hospitals near **{city}, {region}**]({map_url})", unsafe_allow_html=True)
            st.caption("ℹ️ Based on your current location via IP.")
        except:
            st.warning("📡 Unable to fetch location. Please check internet connection.")

    else:
        st.success(f"✅ **Low Risk** of Lung Cancer\n\n🧪 Confidence: `{prob*100:.2f}%`")
        if IS_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(23, GPIO.OUT)
            GPIO.output(23, GPIO.HIGH)
            st.image("https://media.giphy.com/media/3ohzdYJK1wAdPWVk88/giphy.gif", width=280)

    st.caption(f"📌 Model Used: **{model_choice}**")

# GPIO cleanup
if IS_PI:
    import atexit
    atexit.register(GPIO.cleanup)
