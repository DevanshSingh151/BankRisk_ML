import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

API_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="BankRisk360 Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

def check_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

st.title("🏦 BankRisk360")
st.markdown("Intelligent Banking Decision & Risk Analytics Platform")

if not check_health():
    st.error("Backend API is not running. Please start the FastAPI server.")
    st.stop()

# Horizontal Navigation
pages = ["Dashboard", "Customer 360", "Model Insights", "About"]
selection = st.radio("Navigation", pages, horizontal=True, label_visibility="collapsed")
st.markdown("---")

if selection == "Dashboard":
    from frontend.views.dashboard import render_dashboard
    render_dashboard(API_URL)
elif selection == "Customer 360":
    from frontend.views.customer_360 import render_customer_360
    render_customer_360(API_URL)
elif selection == "Model Insights":
    from frontend.views.model_insights import render_model_insights
    render_model_insights(API_URL)
elif selection == "About":
    st.title("About BankRisk360")
    st.markdown("""
    ## Overview
    BankRisk360 is an intelligent decision support system designed for modern banking.
    It demonstrates how machine learning can support decisions through:
    - Alternative Credit Scoring
    - Early Warning Systems
    - Customer Behavioral Segmentation
    - Next-Best-Offer Recommendation
    
    ### Predict → Explain → Recommend
    We do not just output a probability. Every prediction explains *why* the model made that decision using SHAP values, and recommends a potential banking action based on the insights.
    
    ### Dataset
    Trained on the public "Default of Credit Card Clients Dataset" (UCI/OpenML).
    """)
