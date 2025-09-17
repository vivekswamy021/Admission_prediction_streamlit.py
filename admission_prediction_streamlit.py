import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

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
        ax.hist(df[feature], bins=bins, rwidth=0.8)
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    elif analysis_type == "Bivariate Analysis":
        x_feature = st.selectbox("Select X-axis Feature", numeric_cols, index=0)
        y_feature = st.selectbox("Select Y-axis Feature", numeric_cols, index=1)
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

available_models = {
    "Linear Regression": "linear_regression_models.joblib",  # replace with your actual path
    "Lasso Regression": "lasso_models.joblib",
    "Support Vector Regressor (SVR)": "svr_models.joblib",
    "Decision Tree": "decision_tree_models.joblib",
    "Random Forest": "random_forest_models.joblib",
    "K-Nearest Neighbors": "knn_models.joblib"
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
        st.success(f"🎯 Predicted Admission Probability using {model_choice}: **{prediction:.2f}**")
else:
    st.warning(f"⚠️ {selected_model_file} not found. Please train and save this model first.")
