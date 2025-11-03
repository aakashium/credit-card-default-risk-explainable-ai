"""
app.py
------
Streamlit application that integrates:
1. Data Preprocessing
2. Model Training
3. Model Explainability (SHAP)
4. About Section
Each module can be run independently using buttons.
"""

import streamlit as st
from src.data_preprocessing import run_data_preprocessing
from src.model_training import train_model
from src.explainability import run_explainability_pipeline
import time
from logger_config import setup_logger

# Initialize Logger
logger = setup_logger("app_logger", "logs/app.log")

# Streamlit Page Config 
st.set_page_config(page_title="Credit Risk ML Pipeline", layout="wide")

# Sidebar Navigation 
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("Data Preprocessing", "Model Training", "Explainability Analysis", "About"),
)

# App Title 
st.title("Credit Risk Prediction - Modular ML Pipeline")
st.markdown("### Machine Learning + Explainable AI Workflow")
st.divider()

# PAGE 1: DATA PREPROCESSING

if page == "Data Preprocessing":
    st.header("Data Preprocessing")
    st.write("Prepare raw dataset for model training — includes cleaning, encoding, and scaling.")

    if st.button("Run Data Preprocessing"):
        with st.spinner("Running data preprocessing..."):
            try:
                run_data_preprocessing()
                st.success("Data preprocessing completed successfully!")
                logger.info("Data preprocessing executed successfully.")
            except Exception as e:
                st.error(f"Error in preprocessing: {e}")
                logger.exception(f"Error in data preprocessing: {e}")

# PAGE 2: MODEL TRAINING

elif page == "Model Training":
    st.header("Model Training")
    st.write("Train ML model using preprocessed data and evaluate performance.")

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                TRAIN_PATH = "./data/processed/train_preprocessed.csv"
                TEST_PATH = "./data/processed/test_preprocessed.csv"
                TARGET_COL = "DEFAULT"
                model, metrics = train_model(TRAIN_PATH, TEST_PATH, TARGET_COL)
                st.success("Model training completed successfully!")
                st.write("### Model Performance Metrics")
                st.json(metrics)
                logger.info("Model training executed successfully.")
            except Exception as e:
                st.error(f"Error in model training: {e}")
                logger.exception(f"Error in model training: {e}")

# PAGE 3: EXPLAINABILITY ANALYSIS

elif page == "Explainability Analysis":
    st.header("Model Explainability")
    st.write("Generate SHAP-based explainability visuals to interpret model predictions.")
    if "explainability_ran" not in st.session_state:
        st.session_state.explainability_ran = False
    if not st.session_state.explainability_ran:
        if st.button("Run Explainability Analysis", type="primary"):
            st.session_state.explainability_ran = True
        else:
            st.info("Explainability analysis has already been run. Refresh the page to run again.") 
            with st.spinner("Generating SHAP explainability plots.. Have Patience as it may take upto 30 minutes to load..."):
                try:
                    results = run_explainability_pipeline()
                    st.success("Explainability analysis completed successfully!")

                    # Display Plots
                    st.subheader("Global Feature Importance")
                    st.image(results["bar_plot"], caption="SHAP Feature Importance (Bar Plot)")
                    st.image(results["summary_plot"], caption="SHAP Summary Plot")

                    st.subheader("Local Explanation (Sample #10)")
                    st.image(results["force_plot"], caption="SHAP Force Plot")
                    st.image(results["waterfall_plot"], caption="SHAP Waterfall Plot")

                    st.subheader("Dependence Plots (Top Features)")
                    for dep_path in results["dependence_plots"]:
                        st.image(dep_path, caption=f"Dependence Plot: {dep_path.split('/')[-1]}")

                    logger.info("Explainability analysis executed successfully.")
                except Exception as e:
                    st.error(f"Error in explainability analysis: {e}")
                    logger.exception(f"Error in explainability analysis: {e}")               
                

# PAGE 4: ABOUT SECTION

elif page == "About":
    st.header(" About This App")
    st.markdown(
        """
        ### **Credit Default Prediction Dashboard**
        Built with **Streamlit**, **XGBoost**, and **SHAP**  
        This app demonstrates a complete Machine Learning workflow —  
        from **data preprocessing → model training → explainability**.

        #### Features:
        - Modular pipeline for maintainability  
        - Model training using XGBoost Classifier  
        - SHAP-based global and local interpretability  
        - Interactive visualization of results  

        #### Author:
        **Aakash Mohan**

        #### Model:
        - **Algorithm:** XGBoost Classifier  
        - **Explainability Tool:** SHAP (SHapley Additive exPlanations)
        """
    )


st.divider()
st.caption("© 2025 Aakash Mohan — All Rights Reserved")
