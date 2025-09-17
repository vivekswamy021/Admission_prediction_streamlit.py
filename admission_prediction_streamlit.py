import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="🎓 Admission Prediction", layout="wide")

# ------------------------------
# APP TITLE
# ------------------------------
st.title("🎓 Admission Probability Prediction")
st.markdown("""
This app predicts the **probability of admission** based on:  
**GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, and Research Experience**.
""")

# ------------------------------
# STEP 0: LOAD DATA
# ------------------------------
st.sidebar.header("📁 Upload Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload admission dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.success("Dataset uploaded successfully!")

    # Rename columns for consistency
    df = df.rename(columns={
        'GRE Score': 'GRE_Score',
        'TOEFL Score': 'TOEFL_Score',
        'University Rating': 'University_Rating',
        'SOP': 'SOP_Rating',
        'LOR ': 'LOR',
        'Chance of Admit ': 'probability_of_admit'
    })

    # ------------------------------
    # STEP 1: EDA
    # ------------------------------
    st.header("🔎 Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    buffer = df.info()
    st.text(buffer)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Feature Distributions")
    features = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP_Rating', 'LOR', 'CGPA', 'Research']
    for col in features:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=10, rwidth=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    st.subheader("Feature vs Target (Scatter Plots)")
    for col in ['GRE_Score', 'TOEFL_Score', 'CGPA']:
        fig, ax = plt.subplots()
        ax.scatter(df[col], df['probability_of_admit'])
        ax.set_xlabel(col)
        ax.set_ylabel("Probability of Admit")
        ax.set_title(f"{col} vs Probability of Admit")
        st.pyplot(fig)

    # ------------------------------
    # STEP 2: USER INPUT FORM
    # ------------------------------
    st.sidebar.header("📌 Enter Student Details")

    gre = st.sidebar.number_input("GRE Score", 260, 340, 300)
    toefl = st.sidebar.number_input("TOEFL Score", 0, 120, 100)
    uni_rating = st.sidebar.slider("University Rating", 1, 5, 3)
    sop = st.sidebar.slider("SOP Rating", 1.0, 5.0, 3.0)
    lor = st.sidebar.slider("LOR Rating", 1.0, 5.0, 3.0)
    cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0, 8.0)
    research = st.sidebar.selectbox("Research Experience", [0, 1])

    input_features = np.array([[gre, toefl, uni_rating, sop, lor, cgpa, research]])

    # ------------------------------
    # STEP 3: SHOW USER INPUT FEATURES
    # ------------------------------
    st.subheader("📝 Input Features Summary")
    input_dict = {
        "GRE Score": gre,
        "TOEFL Score": toefl,
        "University Rating": uni_rating,
        "SOP Rating": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research
    }
    st.table(input_dict)

    # ------------------------------
    # STEP 4: MODEL SELECTION
    # ------------------------------
    st.sidebar.header("⚙️ Choose Model")

    saved_model_dir = "saved_models"
    available_models = {
        "Linear Regression": os.path.join(saved_model_dir, "linear_regression_model(1).joblib"),
        "Lasso Regression": os.path.join(saved_model_dir, "lasso_model.joblib"),
        "Support Vector Regressor (SVR)": os.path.join(saved_model_dir, "svr_model.joblib"),
        "Decision Tree": os.path.join(saved_model_dir, "decision_tree_model.joblib"),
        "Random Forest": os.path.join(saved_model_dir, "random_forest_model.joblib"),
        "K-Nearest Neighbors": os.path.join(saved_model_dir, "knn_model.joblib")
    }

    model_choice = st.sidebar.selectbox("Select Model", list(available_models.keys()))
    selected_model_file = available_models[model_choice]

    # ------------------------------
    # STEP 5: LOAD MODEL & PREDICT
    # ------------------------------
    st.header("📂 Prediction")

    if os.path.exists(selected_model_file):
        @st.cache_resource
        def load_model(file_path):
            return joblib.load(file_path)

        model = load_model(selected_model_file)

        if st.button("🔮 Predict Admission Probability"):
            prediction = model.predict(input_features)[0]
            st.success(f"🎯 Predicted Admission Probability using {model_choice}: **{prediction*100:.2f}%**")
    else:
        st.warning(f"⚠️ {selected_model_file} not found. Please train and save this model first.")
else:
    st.info("Please upload the admission dataset CSV to start EDA and predictions.")
