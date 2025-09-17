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

st.title("🎓 Admission Probability Prediction & EDA")
st.markdown("""
Upload the admission dataset to explore it with **Univariate, Bivariate, and Multivariate analysis**,  
and predict the probability of admission for a candidate.
""")

# ------------------------------
# STEP 0: UPLOAD DATA
# ------------------------------
st.sidebar.header("📁 Upload Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload admission dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Rename columns for consistency
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
    # INTERACTIVE EDA
    # ------------------------------
    st.sidebar.header("📊 EDA Options")
    analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                         ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

    st.header(f"📈 {analysis_type}")

    numeric_cols = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP_Rating', 'LOR', 'CGPA', 'Research']

    if analysis_type == "Univariate Analysis":
        feature = st.selectbox("Select Feature for Histogram", numeric_cols)
        bins = st.slider("Number of bins", 5, 50, 10)
        fig, ax = plt.subplots()
        ax.hist(df[feature], bins=bins, rwidth=0.8)
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif analysis_type == "Bivariate Analysis":
        x_feature = st.selectbox("Select X-axis Feature", numeric_cols)
        y_feature = st.selectbox("Select Y-axis Feature", numeric_cols)
        fig, ax = plt.subplots()
        ax.scatter(df[x_feature], df[y_feature])
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f"{x_feature} vs {y_feature}")
        st.pyplot(fig)

    elif analysis_type == "Multivariate Analysis":
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Pairplot (Scatterplot Matrix)")
        pairplot_fig = sns.pairplot(df[numeric_cols])
        st.pyplot(pairplot_fig.fig)

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

    input_features = np.array([[gre, toefl, uni_rating, sop, lor, cgpa, research]])

    # Show input summary
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
    # STEP 2: MODEL SELECTION
    # ------------------------------
    st.sidebar.header("⚙️ Choose Model")
    saved_model_dir = "saved_models"  # folder where your joblib models are stored

    saved_model_dir = "/content/saved_models"

available_models = {
    "Linear Regression": os.path.join(saved_model_dir, "linear_regression_models.joblib"),
    "Lasso Regression": os.path.join(saved_model_dir, "lasso_model.joblib"),
    "SVR": os.path.join(saved_model_dir, "svr_model.joblib"),
    "Decision Tree": os.path.join(saved_model_dir, "decision_tree_model.joblib"),
    "Random Forest": os.path.join(saved_model_dir, "random_forest_model.joblib"),
    "KNN": os.path.join(saved_model_dir, "knn_model.joblib")}

model_choice = st.sidebar.selectbox("Select Model", list(available_models.keys()))
selected_model_file = available_models[model_choice]

# ------------------------------
# STEP 3: LOAD MODEL & PREDICT
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
