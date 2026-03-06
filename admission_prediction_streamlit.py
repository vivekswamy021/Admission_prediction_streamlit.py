import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="🎓 Admission Prediction", layout="wide")

# ------------------------------
# APP TITLE
# ------------------------------
st.title("🎓 Admission Probability Prediction & EDA")
st.markdown("""
This app allows you to explore the dataset and predicts the **probability of admission** based on  
**GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, and Research Experience**.
""")

# ------------------------------
# STEP 0: UPLOAD DATA
# ------------------------------
st.sidebar.header("📁 Upload Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload admission dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Rename columns for consistency (handling potential trailing spaces in CSV)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        'GRE Score': 'GRE_Score',
        'TOEFL Score': 'TOEFL_Score',
        'University Rating': 'University_Rating',
        'SOP': 'SOP_Rating',
        'LOR ': 'LOR',
        'Chance of Admit ': 'probability_of_admit'
    })

    st.header("🔎 Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # EDA SELECTBOX
    # ------------------------------
    st.sidebar.header("📊 EDA Options")
    analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                         ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
    
    numeric_cols = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP_Rating', 'LOR', 'CGPA', 'Research']

    st.header(f"📈 {analysis_type}")
    
    if analysis_type == "Univariate Analysis":
        feature = st.selectbox("Select Feature for Histogram", numeric_cols)
        bins = st.slider("Number of bins", 5, 50, 10)
        fig, ax = plt.subplots()
        ax.hist(df[feature].dropna(), bins=bins, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)
    
    elif analysis_type == "Bivariate Analysis":
        x_feature = st.selectbox("Select X-axis Feature", numeric_cols, index=0)
        y_feature = st.selectbox("Select Y-axis Feature", numeric_cols, index=5)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_feature, y=y_feature, ax=ax)
        ax.set_title(f"{x_feature} vs {y_feature}")
        st.pyplot(fig)
    
    elif analysis_type == "Multivariate Analysis":
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ------------------------------
# STEP 1: USER INPUT FORM
# ------------------------------
st.sidebar.header("📌 Enter Student Details")

gre = st.sidebar.number_input("GRE Score", 260, 340, 300)
toefl = st.sidebar.number_input("TOEFL Score", 0, 120, 100)
uni_rating = st.sidebar.slider("University Rating", 1, 5, 3)
sop = st.sidebar.slider("SOP Rating", 1.0, 5.0, 3.0)
lor = st.sidebar.slider("LOR Rating", 1.0, 5.0, 3.0)
cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0, 8.0)
research = st.sidebar.selectbox("Research Experience", [0, 1])

# Match the feature order used during training
input_features = np.array([[gre, toefl, uni_rating, sop, lor, cgpa, research]])

st.subheader("📝 Input Features Summary")
input_df = pd.DataFrame([input_features[0]], columns=["GRE", "TOEFL", "Uni Rating", "SOP", "LOR", "CGPA", "Research"])
st.table(input_df)

# ------------------------------
# STEP 2: MODEL SELECTION
# ------------------------------
st.sidebar.header("⚙️ Choose Model")

available_models = {
    "Linear Regression": "linear_regression_models.joblib",
    "Lasso Regression": "lasso_models.joblib",
    "Random Forest": "random_forest_models.joblib"
}

model_choice = st.sidebar.selectbox("Select Model", list(available_models.keys()))
selected_model_file = available_models[model_choice]

# ------------------------------
# STEP 3: LOAD MODEL & PREDICT
# ------------------------------
st.header("📂 Prediction")

if os.path.exists(selected_model_file):
    model = joblib.load(selected_model_file)
    if st.button("🔮 Predict Admission Probability"):
        prediction = model.predict(input_features)[0]
        # Ensure prediction is within 0-1 range
        prediction = max(0, min(1, prediction))
        st.success(f"🎯 Predicted Admission Probability using {model_choice}: **{prediction:.2f}**")
else:
    st.error(f"⚠️ Model file '{selected_model_file}' not found in the repository.")
    st.info("Make sure you have uploaded your .joblib files to GitHub along with this script.")
