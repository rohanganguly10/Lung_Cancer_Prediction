# -----------------------------
# 🏥 Introduction Page (pages/1_🏥_Introduction.py)
# -----------------------------
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Introduction", page_icon="🏥", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #008080;'>Lung Cancer Prediction and SDG-3 Impact</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI for Good: Supporting Sustainable Development Goal 3 – Good Health and Well-being</h4>", unsafe_allow_html=True)
st.markdown("---")

# Two-column layout with image and text
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📌 Why This Matters:")
    st.markdown("""
    Lung cancer is one of the most aggressive and fatal diseases, especially when not detected early.
    
    - 🔬 **This app uses advanced machine learning algorithms** to assess lung cancer risk based on symptoms and health history.
    - 🧠 Models used include **Random Forest, XGBoost**, and others evaluated across accuracy and performance metrics.
    - 🎯 Built using real patient data, cleaned and engineered with 16 health-related features.
    - 🤖 Deployed for accessibility on **web and Raspberry Pi** for real-time screening anywhere.
    
    ### 🌍 SDG-3 Goals Aligned:
    - 🚑 **Goal 3.4 of SDG-3** aims to reduce premature mortality from non-communicable diseases (NCDs) — including cancer — by **one-third** by the year **2030** through prevention, early detection, and treatment.
    - 🩺 According to WHO, lung cancer accounts for **1 in 5 cancer deaths** globally. Late detection significantly lowers survival rates.
    - 🧭 This app empowers **frontline health workers and educators** with real-time prediction tools, removing the dependency on urban diagnostics.
    > 🎯 *This project contributes directly to SDG Target 3.4 and 3.d, which call for enhanced capacity of health systems, particularly in developing countries.*
""")

with col2:
    try:
        image = Image.open("lungs_new.png")
        st.image(image, use_container_width=True)
    except:
        st.warning("Please ensure 'lungs.png' exists in your working directory.")