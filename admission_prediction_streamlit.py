import streamlit as st
import numpy as np
import joblib
import os
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="🎓 Admission Prediction", layout="wide")

# ------------------------------
# APP TITLE
# ------------------------------
st.title("🎓 Admission Probability Prediction")
st.markdown("""
This app predicts the **probability of admission** based on  
**GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, and Research Experience**.
""")

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

# ------------------------------
# STEP 2: MODEL SELECTION
# ------------------------------
st.sidebar.header("⚙️ Choose Model")

available_models = {
    "Linear Regression": "linear_regression_model(1).joblib",
    "Lasso Regression": "lasso_model.joblib",
    "Support Vector Regressor (SVR)": "svr_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "K-Nearest Neighbors": "knn_model.joblib"
}

model_choice = st.sidebar.selectbox("Select Model", list(available_models.keys()))

# ------------------------------
# STEP 3: LOAD MODEL & PREDICT
# ------------------------------
st.header("📂 Prediction")

selected_model_file = os.path.join("saved_models", available_models[model_choice])

if os.path.exists(selected_model_file):
    model = joblib.load(selected_model_file)

    if st.button("🔮 Predict Admission Probability"):
        prediction = model.predict(input_features)[0]
        st.success(f"🎯 Predicted Admission Probability using {model_choice}: **{prediction:.2f}**")
else:
    st.warning(f"⚠️ {available_models[model_choice]} not found. Please train and save this model first.")

# ------------------------------
# STEP 4: DOWNLOAD MODELS
# ------------------------------
st.subheader("📥 Download Trained Models")
models_dir = "saved_models"
if os.path.exists(models_dir):
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), "rb") as f:
            st.download_button(
                label=f"Download {model_file}",
                data=f,
                file_name=model_file,
                mime="application/octet-stream"
            )
