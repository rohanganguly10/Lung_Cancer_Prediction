# -----------------------------
# 📊 Data Visualization Page (pages/2_📊_Data_Visualization.py)
# -----------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Visualization", layout="wide")

# ----------------- Title -----------------
st.markdown("<h1 style='text-align:center; color:#008080;'>Data Insights & Visualization</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Understanding Feature corelation with lung cancer risk</h4>", unsafe_allow_html=True)
st.markdown("---")

# ----------------- Load Data -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("survey lung cancer.csv")
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
    return df

df = load_data()

# ----------------- Raw Data Section -----------------
st.markdown("### 🗃️ Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("🧾 Records", len(df))
col2.metric("🧬 Features", len(df.columns) - 1)
col3.metric("💡 Lung Cancer Positive (%)", f"{(df['LUNG_CANCER'].value_counts(normalize=True)['YES']*100):.1f}%")

with st.expander("🔍 Click to view raw dataset"):
    st.dataframe(df, height=250)
    st.caption("📝 Data collected from real-world lung cancer surveys. All responses are anonymized.")

st.markdown("### 📊 Visual Comparison of Features")

selected_feature = st.selectbox("🔎 Choose a Feature to Explore", df.columns.drop("LUNG_CANCER"))
feature_type = df[selected_feature].dtype

# Auto plot
if feature_type == "object":
    # Auto use stacked bar

    grouped = df.groupby([selected_feature, "LUNG_CANCER"]).size().reset_index(name='count')
    total = grouped.groupby(selected_feature)['count'].transform('sum')
    grouped['percentage'] = grouped['count'] / total * 100

    fig = px.bar(grouped, x=selected_feature, y="percentage", color="LUNG_CANCER",
                 barmode="stack", text_auto=".1f",
                 color_discrete_sequence=["#EF553B", "#00CC96"])
    fig.update_layout(yaxis_title="Percentage", title=f"{selected_feature} Distribution by Lung Cancer")
    st.plotly_chart(fig, use_container_width=True)

else:
    unique_vals = df[selected_feature].nunique()

    if unique_vals > 5:
        
        fig = px.box(df, x="LUNG_CANCER", y=selected_feature, color="LUNG_CANCER",
                     color_discrete_sequence=["#EF553B", "#00CC96"])
    else:
        
        means = df.groupby("LUNG_CANCER")[selected_feature].mean().reset_index()
        fig = px.bar(means, x="LUNG_CANCER", y=selected_feature, color="LUNG_CANCER",
                     text_auto=".2f", color_discrete_sequence=["#EF553B", "#00CC96"])
        fig.update_layout(title=f"Mean {selected_feature} by Lung Cancer Class")

    st.plotly_chart(fig, use_container_width=True)


# ----------------- Correlation Heatmap -----------------
st.markdown("### 🔥 Correlation Matrix")
with st.expander("Show Heatmap of Feature Correlations"):
    encoded_df = df.copy()
    for col in encoded_df.select_dtypes("object").columns:
        encoded_df[col] = encoded_df[col].astype("category").cat.codes

    corr = encoded_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.3, ax=ax)
    st.pyplot(fig)
    st.caption("📌 Note: Anxiety × Yellow Fingers and Chronic Disease show strong correlations with Lung Cancer.")

with st.expander("🔥 Top 5 Correlated Features with Lung Cancer"):
    enc_df = df.copy()
    for col in enc_df.select_dtypes("object").columns:
        enc_df[col] = enc_df[col].astype("category").cat.codes
    correlations = enc_df.corr()["LUNG_CANCER"].drop("LUNG_CANCER").abs().sort_values(ascending=False).head(5)
    st.bar_chart(correlations)
    st.caption("These features had the strongest linear correlation with lung cancer outcome.")


# ----------------- Footer -----------------
st.markdown("---")
st.success("✅ These insights helped select features for model training.\nExplore predictions in the 🧠 **Lung Cancer Predictor**.")
