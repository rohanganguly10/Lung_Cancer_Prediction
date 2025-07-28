# app.py
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Lung Cancer Dashboard",
    page_icon="🫁",
    layout="wide"
)

st.markdown("<h1 style='text-align:center; color:#008080;'>🫁 Lung Cancer Risk Assessment Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Use the sidebar to navigate across sections →</h4>", unsafe_allow_html=True)
st.markdown("---")


st.markdown("## 🚀 Welcome!")
st.info("""
This dashboard includes:
- 🏥 Introduction to the SDG-3 aligned project
- 📊 Data Exploration and Feature Impact
- 🧠 Interactive Lung Cancer Risk Prediction Tool

Use the **sidebar** on the left to explore each module.
""")
