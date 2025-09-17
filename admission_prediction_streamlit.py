import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="🎓 Admission Prediction", layout="wide")

# ------------------------------
# STEP 1: BUSINESS PROBLEM
# ------------------------------
st.title("🎓 Admission Probability Prediction")
st.markdown("""
This app predicts the **probability of admission** for a student applying to an institution,  
based on features such as **GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, and Research Experience**.
""")

# ------------------------------
# STEP 2: LOAD DATA
# ------------------------------
st.header("📂 Load & Understand Data")

@st.cache_data
def load_data():
    df = pd.read_csv("data/admission_predict.csv")
    df = df.rename(columns={
        'Chance of Admit ': 'probability_of_admit',
        'GRE Score': 'GRE_Score',
        'TOEFL Score': 'TOEFL_Score',
        'University Rating': 'University_Rating',
        'SOP': 'SOP_Rating',
        'LOR ': 'LOR'
    })
    df.drop(columns=['Serial No.'], inplace=True)
    return df

df = load_data()
st.dataframe(df.head())

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Target", "probability_of_admit")

# ------------------------------
# STEP 3: DATA EXPLORATION
# ------------------------------
st.header("📊 Exploratory Data Analysis")

if st.checkbox("Show Data Info"):
    buffer = []
    df.info(buf=buffer)
    s = "\n".join(buffer)
    st.text(s)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ------------------------------
# STEP 4: FEATURES & SPLIT
# ------------------------------
st.header("⚙️ Model Training")

# Replace 0 with NaN for continuous vars
df_copy = df.copy()
df_copy[['GRE_Score','TOEFL_Score','University_Rating','SOP_Rating','LOR','CGPA']] = df_copy[['GRE_Score','TOEFL_Score','University_Rating','SOP_Rating','LOR','CGPA']].replace(0, np.nan)

X = df_copy.drop(columns=['probability_of_admit'])
y = df_copy['probability_of_admit']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# STEP 5: TRAIN MULTIPLE MODELS
# ------------------------------
def find_best_model_and_save(X, y, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    models = {
        'linear_regression': {'model': LinearRegression(), 'parameters': {}},
        'lasso': {'model': Lasso(), 'parameters': {'alpha': [1, 2], 'selection': ['random', 'cyclic']}},
        'svr': {'model': SVR(), 'parameters': {'gamma': ['auto','scale']}},
        'decision_tree': {'model': DecisionTreeRegressor(), 'parameters': {'criterion': ['squared_error','friedman_mse'], 'splitter': ['best','random']}},
        'random_forest': {'model': RandomForestRegressor(criterion='squared_error'), 'parameters': {'n_estimators':[5,10,15,50]}},
        'knn': {'model': KNeighborsRegressor(algorithm='auto'), 'parameters': {'n_neighbors':[2,5,10,20]}}
    }

    scores, best_estimators = [], {}

    for model_name, config in models.items():
        gs = GridSearchCV(config['model'], config['parameters'], cv=5, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
        best_estimators[model_name] = gs.best_estimator_
        joblib.dump(gs.best_estimator_, os.path.join(save_dir, f"{model_name}_model.joblib"))

    results_df = pd.DataFrame(scores)
    best_model_name = results_df.loc[results_df['best_score'].idxmax(), 'model']
    best_model = best_estimators[best_model_name]
    return results_df, best_model_name, best_model

if st.button("🔍 Train & Compare Models"):
    results_df, best_model_name, best_model = find_best_model_and_save(X, y)
    st.write("### Model Comparison")
    st.dataframe(results_df)

    # Evaluate Best Model
    y_pred = best_model.predict(x_test)
    st.write(f"✅ **Best Model:** {best_model_name}")
    st.write("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("Test R²:", r2_score(y_test, y_pred))
    st.write("Cross Val Score:", cross_val_score(best_model, X, y, cv=5).mean())

    # ------------------------------
    # STEP 6: DOWNLOAD MODELS
    # ------------------------------
    st.subheader("📥 Download Trained Models")
    models_dir = "saved_models"
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), "rb") as f:
            st.download_button(
                label=f"Download {model_file}",
                data=f,
                file_name=model_file,
                mime="application/octet-stream"
            )
