import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="🎓 Admission Prediction", layout="wide")

# --- APP TITLE ---
st.title("🎓 Admission Probability Prediction & EDA")
st.markdown("""
This app predicts the **probability of admission** based on academic parameters and provides exploratory data analysis.
""")

# --- STEP 0: UPLOAD DATA ---
st.sidebar.header("📁 1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload admission dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Cleaning column names (handling trailing spaces found in common admission datasets)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        'GRE Score': 'GRE_Score',
        'TOEFL Score': 'TOEFL_Score',
        'University Rating': 'University_Rating',
        'SOP': 'SOP_Rating',
        'LOR': 'LOR',
        'Chance of Admit': 'probability_of_admit'
    })

    st.header("🔎 Dataset Preview")
    st.dataframe(df.head())

    # --- EDA SECTION ---
    st.sidebar.header("📊 2. EDA Options")
    analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                         ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
    
    numeric_cols = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP_Rating', 'LOR', 'CGPA', 'Research']

    st.header(f"📈 {analysis_type}")
    
    if analysis_type == "Univariate Analysis":
        feature = st.selectbox("Select Feature for Histogram", numeric_cols)
        bins = st.slider("Number of bins", 5, 50, 20)
        fig, ax = plt.subplots()
        sns.histplot(df[feature], bins=bins, kde=True, ax=ax, color='skyblue')
        st.pyplot(fig)
    
    elif analysis_type == "Bivariate Analysis":
        x_feature = st.selectbox("Select X-axis", numeric_cols, index=0)
        y_feature = st.selectbox("Select Y-axis", numeric_cols, index=5)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='Research', ax=ax)
        st.pyplot(fig)
    
    elif analysis_type == "Multivariate Analysis":
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# --- STEP 1: USER INPUT FORM ---
st.sidebar.header("📌 3. Student Details")

gre = st.sidebar.number_input("GRE Score", 260, 340, 310)
toefl = st.sidebar.number_input("TOEFL Score", 0, 120, 105)
uni_rating = st.sidebar.slider("University Rating", 1, 5, 3)
sop = st.sidebar.slider("SOP Rating", 1.0, 5.0, 3.5)
lor = st.sidebar.slider("LOR Rating", 1.0, 5.0, 3.5)
cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0, 8.5)
research = st.sidebar.selectbox("Research Experience (0=No, 1=Yes)", [0, 1])

# Prepare input for model
input_data = np.array([[gre, toefl, uni_rating, sop, lor, cgpa, research]])

# --- STEP 2: MODEL SELECTION ---
st.sidebar.header("⚙️ 4. Model Selection")

available_models = {
    "Linear Regression": "linear_regression_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "SVR": "svr_model.joblib"
}

model_choice = st.sidebar.selectbox("Choose Model", list(available_models.keys()))
model_path = available_models[model_choice]

# --- STEP 3: PREDICTION ---
st.header("🔮 Prediction Results")

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        if st.button("Calculate Admission Probability"):
            prediction = model.predict(input_data)[0]
            # Clip result between 0 and 1
            res = max(0, min(1, prediction))
            st.success(f"### Predicted Chance: **{res*100:.2f}%**")
            st.progress(res)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.info(f"💡 Model file `{model_path}` not found in repository. Using a simple mock formula for demo purposes.")
    if st.button("Calculate (Mock Mode)"):
        # Mock logic: (CGPA/10 * 0.7) + (GRE/340 * 0.3)
        mock_res = (cgpa / 10 * 0.7) + (gre / 340 * 0.3)
        st.warning(f"### Mock Predicted Chance: **{mock_res*100:.2f}%**")
