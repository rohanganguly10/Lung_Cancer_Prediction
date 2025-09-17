import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
from PIL import Image
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

# Page setup
st.set_page_config(page_title="Lung Cancer App", page_icon="🏥", layout="wide")

# -----------------------------
# Tabs for sections
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏥 Introduction", "📊 Data Visualization", "🫁 Lung Cancer Predictor"])

# -----------------------------
# Tab 1: Introduction
# -----------------------------
with tab1:
    st.markdown("<h1 style='text-align: center; color: #008080;'>Lung Cancer Prediction and SDG-3 Impact</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>AI for Good: Supporting SDG-3 – Good Health and Well-being</h4>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown("### 📌 Why This Matters:")
        st.markdown("""
        Lung cancer is one of the most aggressive and fatal diseases.
        - 🔬 **Advanced ML models** assess lung cancer risk.
        - 🧠 Models: Random Forest, XGBoost, etc.
        - 🎯 Real patient data with 16 features.
        - 🤖 Deployed on web & Raspberry Pi.
        
        ### 🌍 SDG-3 Goals:
        - 🚑 Reduce premature mortality from NCDs by **1/3 by 2030**.
        - 🩺 Lung cancer = **1 in 5 cancer deaths** globally.
        - 🧭 Supports frontline health workers with real-time predictions.
        > 🎯 Contributes to SDG Targets 3.4 & 3.d
        """)

    with col2:
        try:
            image = Image.open("lungs_new.png")
            st.image(image, use_container_width=True)
        except:
            st.warning("Please ensure 'lungs_new.png' exists in your directory.")

# -----------------------------
# Tab 2: Data Visualization
# -----------------------------
with tab2:
    st.markdown("<h1 style='text-align:center; color:#008080;'>Data Insights & Visualization</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Understanding Feature correlation with lung cancer risk</h4>", unsafe_allow_html=True)
    st.markdown("---")

    @st.cache_data
    def load_data():
        df = pd.read_csv("survey lung cancer.csv")
        df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
        return df

    df = load_data()

    st.markdown("### 🗃️ Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧾 Records", len(df))
    col2.metric("🧬 Features", len(df.columns)-1)
    col3.metric("💡 Lung Cancer Positive (%)", f"{(df['LUNG_CANCER'].value_counts(normalize=True)['YES']*100):.1f}%")

    with st.expander("🔍 Click to view raw dataset"):
        st.dataframe(df, height=250)
        st.caption("📝 Anonymized survey data")

    st.markdown("### 📊 Visual Comparison of Features")
    selected_feature = st.selectbox("🔎 Choose a Feature to Explore", df.columns.drop("LUNG_CANCER"))

    if df[selected_feature].dtype == "object":
        grouped = df.groupby([selected_feature, "LUNG_CANCER"]).size().reset_index(name='count')
        total = grouped.groupby(selected_feature)['count'].transform('sum')
        grouped['percentage'] = grouped['count']/total*100
        fig = px.bar(grouped, x=selected_feature, y="percentage", color="LUNG_CANCER", barmode="stack",
                     text_auto=".1f", color_discrete_sequence=["#EF553B","#00CC96"])
        fig.update_layout(yaxis_title="Percentage")
    else:
        unique_vals = df[selected_feature].nunique()
        if unique_vals > 5:
            fig = px.box(df, x="LUNG_CANCER", y=selected_feature, color="LUNG_CANCER",
                         color_discrete_sequence=["#EF553B","#00CC96"])
        else:
            means = df.groupby("LUNG_CANCER")[selected_feature].mean().reset_index()
            fig = px.bar(means, x="LUNG_CANCER", y=selected_feature, color="LUNG_CANCER",
                         text_auto=".2f", color_discrete_sequence=["#EF553B","#00CC96"])
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("### 🔥 Correlation Matrix")
    with st.expander("Show Heatmap of Feature Correlations"):
        enc_df = df.copy()
        for col in enc_df.select_dtypes("object").columns:
            enc_df[col] = enc_df[col].astype("category").cat.codes
        corr = enc_df.corr()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.3, ax=ax)
        st.pyplot(fig)

# -----------------------------
# Tab 3: Lung Cancer Predictor
# -----------------------------
with tab3:
    colored_header(
        label="🫁 Lung Cancer Risk Predictor",
        description="AI-powered Screening Tool",
        color_name="green-70",
    )

    st.sidebar.markdown("### ⚙️ Choose ML Model")
    model_choice = st.sidebar.selectbox("Prediction Model:", ["Random Forest", "XGBoost"])

    with st.form("prediction_form"):
        st.markdown("<h4>📋 Enter Patient Symptoms</h4>", unsafe_allow_html=True)
        patient_name = st.text_input("👤 Patient Name")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male","Female"])
            age = st.slider("Age", 20, 90, 50)
            smoking = st.radio("Smoking?", ["Yes","No"])
            yellow_fingers = st.radio("Yellow Fingers?", ["Yes","No"])
            anxiety = st.radio("Anxiety?", ["Yes","No"])
        with col2:
            peer_pressure = st.radio("Peer Pressure?", ["Yes","No"])
            alcohol = st.radio("Alcohol?", ["Yes","No"])
            chronic = st.radio("Chronic Disease?", ["Yes","No"])
            fatigue = st.radio("Fatigue?", ["Yes","No"])
            allergy = st.radio("Allergy?", ["Yes","No"])
        with col3:
            wheezing = st.radio("Wheezing?", ["Yes","No"])
            coughing = st.radio("Coughing?", ["Yes","No"])
            breath = st.radio("Shortness of Breath?", ["Yes","No"])
            swallowing = st.radio("Swallowing Difficulty?", ["Yes","No"])
            chest_pain = st.radio("Chest Pain?", ["Yes","No"])
        submitted = st.form_submit_button("🔍 Predict Risk")

    # Encode helper
    encode = lambda x: 1 if x=="Yes" else 0

    if submitted and patient_name:
        features = [
            encode(yellow_fingers), encode(anxiety), encode(peer_pressure),
            encode(chronic), encode(fatigue), encode(allergy), encode(wheezing),
            encode(alcohol), encode(coughing), encode(swallowing), encode(chest_pain)
        ]
        derived = features[0]*features[1]
        final_input = np.array([features + [derived]])

        model = rf_model if model_choice=="Random Forest" else xgb_model
        pred = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0][pred]

        st.markdown("---")
        st.subheader(f"🎯 Prediction Result for **{patient_name}**")
        if pred==1:
            st.error(f"🚨 High Risk of Lung Cancer\nConfidence: {prob*100:.2f}%")
        else:
            st.success(f"✅ Low Risk of Lung Cancer\nConfidence: {prob*100:.2f}%")
