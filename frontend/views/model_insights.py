import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json

def render_model_insights(api_url):
    st.title("🧠 Model Insights & Explainability")
    
    with st.spinner("Loading model metrics..."):
        try:
            res = requests.get(f"{api_url}/model/metrics")
            if res.status_code == 200:
                metrics = res.json()
            else:
                st.error("Could not fetch metrics.")
                return
        except Exception as e:
            st.error(f"Error: {e}")
            return
            
    cr_metrics = metrics.get('credit_risk', {})
    
    st.header("Credit Risk Model: XGBoost")
    st.markdown("We trained an XGBClassifier on the historical dataset to predict default payment next month.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{cr_metrics.get('roc_auc', 0):.3f}")
    col2.metric("PR-AUC", f"{cr_metrics.get('pr_auc', 0):.3f}")
    col3.metric("F1-Score", f"{cr_metrics.get('f1', 0):.3f}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Accuracy", f"{cr_metrics.get('accuracy', 0):.3f}")
    col5.metric("Precision", f"{cr_metrics.get('precision', 0):.3f}")
    col6.metric("Recall", f"{cr_metrics.get('recall', 0):.3f}")
    
    st.markdown("---")
    
    st.subheader("Model Parameters")
    st.code("""
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
    """, language="python")
    
    st.subheader("Class Imbalance Handling")
    st.markdown("Since default datasets are typically imbalanced, we applied **SMOTE** (Synthetic Minority Over-sampling Technique) strictly on the **training data** (after the train/test split) to prevent data leakage and ensure robust evaluation on real hold-out distributions.")
    
    st.subheader("Why XGBoost?")
    st.markdown("""
    - Handles non-linear relationships well.
    - Robust to outliers.
    - Provides native feature importance.
    - Excellent integration with SHAP for individual-level explainability.
    """)

