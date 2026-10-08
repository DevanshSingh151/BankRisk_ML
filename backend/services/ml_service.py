import os
import joblib
import pandas as pd
import numpy as np
import shap

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
models_dir = os.path.join(base_dir, 'models')
data_path = os.path.join(base_dir, 'data', 'processed_data.csv')

# Load models and artifacts lazily
_artifacts = {}

def load_artifacts():
    if not _artifacts:
        print("Loading ML models and artifacts...")
        _artifacts['df'] = pd.read_csv(data_path)
        
        _artifacts['credit_xgb'] = joblib.load(os.path.join(models_dir, 'credit_xgb.joblib'))
        _artifacts['credit_scaler'] = joblib.load(os.path.join(models_dir, 'credit_scaler.joblib'))
        _artifacts['credit_features'] = joblib.load(os.path.join(models_dir, 'credit_features.joblib'))
        _artifacts['credit_metrics'] = joblib.load(os.path.join(models_dir, 'credit_metrics.joblib'))
        
        _artifacts['delinq_xgb'] = joblib.load(os.path.join(models_dir, 'delinquency_xgb.joblib'))
        _artifacts['delinq_scaler'] = joblib.load(os.path.join(models_dir, 'delinquency_scaler.joblib'))
        _artifacts['delinq_features'] = joblib.load(os.path.join(models_dir, 'delinquency_features.joblib'))
        
        _artifacts['kmeans'] = joblib.load(os.path.join(models_dir, 'kmeans.joblib'))
        _artifacts['seg_scaler'] = joblib.load(os.path.join(models_dir, 'segmentation_scaler.joblib'))
        _artifacts['seg_features'] = joblib.load(os.path.join(models_dir, 'segmentation_features.joblib'))
        
        # Initialize explainer
        _artifacts['explainer'] = shap.TreeExplainer(_artifacts['credit_xgb'])
        
    return _artifacts

def get_customer_data(customer_id: int):
    artifacts = load_artifacts()
    df = artifacts['df']
    # The dataset might not have a customer ID, so we use index + 1 as ID
    if customer_id < 1 or customer_id > len(df):
        return None
    return df.iloc[customer_id - 1]

def get_all_customers_info():
    artifacts = load_artifacts()
    return {
        "total": len(artifacts['df']),
        "default_rate": artifacts['df']['default_payment_next_month'].mean()
    }

def get_credit_risk(customer_id: int):
    artifacts = load_artifacts()
    cust_data = get_customer_data(customer_id)
    if cust_data is None: return None
    
    features = artifacts['credit_features']
    X = pd.DataFrame([cust_data[features]])
    
    X_scaled = artifacts['credit_scaler'].transform(X)
    prob = artifacts['credit_xgb'].predict_proba(X_scaled)[0, 1]
    
    score = int((1 - prob) * 100) # 100 is best, 0 is worst
    
    if score >= 75:
        category = "LOW"
        action = "Consider suitable credit/product opportunities, subject to policy and human review."
    elif score >= 50:
        category = "MEDIUM"
        action = "Monitor financial behaviour and consider additional verification before credit decisions."
    else:
        category = "HIGH"
        action = "Prioritize account review and potential customer assistance/intervention."
        
    return {
        "customer_id": customer_id,
        "probability": float(prob),
        "risk_score": score,
        "risk_category": category,
        "action": action
    }

def get_credit_explanation(customer_id: int):
    artifacts = load_artifacts()
    cust_data = get_customer_data(customer_id)
    if cust_data is None: return None
    
    features = artifacts['credit_features']
    X = pd.DataFrame([cust_data[features]])
    X_scaled = artifacts['credit_scaler'].transform(X)
    
    shap_values = artifacts['explainer'].shap_values(X_scaled)
    # Binary classification, shap_values might be a list or array
    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    else:
        contributions = shap_values[0]
        
    base_value = artifacts['explainer'].expected_value
    if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
        base_value = base_value[-1]
        
    feature_contributions = []
    for i, col in enumerate(features):
        feature_contributions.append({
            "feature": col,
            "value": float(X.iloc[0, i]),
            "contribution": float(contributions[i])
        })
        
    # Sort by absolute contribution
    feature_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    
    drivers = [f for f in feature_contributions if f["contribution"] > 0][:5]
    reducers = [f for f in feature_contributions if f["contribution"] < 0][:5]
    
    return {
        "customer_id": customer_id,
        "top_risk_drivers": drivers,
        "top_risk_reducers": reducers,
        "base_value": float(base_value)
    }

def get_delinquency_risk(customer_id: int):
    artifacts = load_artifacts()
    cust_data = get_customer_data(customer_id)
    if cust_data is None: return None
    
    features = artifacts['delinq_features']
    X = pd.DataFrame([cust_data[features]])
    X_scaled = artifacts['delinq_scaler'].transform(X)
    
    prob = artifacts['delinq_xgb'].predict_proba(X_scaled)[0, 1]
    
    score = int(prob * 100)
    if score >= 50:
        category = "HIGH"
        action = "Early customer review / repayment assistance recommended."
    elif score >= 20:
        category = "MEDIUM"
        action = "Monitor short-term repayment patterns."
    else:
        category = "LOW"
        action = "No immediate action required."
        
    return {
        "customer_id": customer_id,
        "probability": float(prob),
        "risk_score": score, # Here higher score = higher risk
        "risk_category": category,
        "action": action
    }

def get_segmentation(customer_id: int):
    artifacts = load_artifacts()
    cust_data = get_customer_data(customer_id)
    if cust_data is None: return None
    
    features = artifacts['seg_features']
    X = pd.DataFrame([cust_data[features]])
    X_scaled = artifacts['seg_scaler'].transform(X)
    
    segment_id = int(artifacts['kmeans'].predict(X_scaled)[0])
    
    # Assign names based on cluster characteristics (heuristic based on common Credit Card behavior)
    # We will dynamically label them based on the cluster centers
    centers = artifacts['kmeans'].cluster_centers_
    # Inverse transform to get actual feature values of centers
    real_centers = artifacts['seg_scaler'].inverse_transform(centers)
    
    # Find the average utilization across clusters
    utilizations = real_centers[:, 2]  # index of avg_utilization
    delays = real_centers[:, 4]        # index of max_delay
    
    names = {}
    for i in range(len(centers)):
        if delays[i] > 1:
            names[i] = "Struggling Borrowers"
        elif utilizations[i] > 0.7:
            names[i] = "High-Utilization Borrowers"
        elif utilizations[i] < 0.2:
            names[i] = "Low-Engagement Savers"
        else:
            names[i] = "Active Balanced Users"
            
    # Guarantee uniqueness just in case
    segment_name = names.get(segment_id, f"Cluster {segment_id}")
    
    chars = {f: float(X.iloc[0, i]) for i, f in enumerate(features)}
    
    return {
        "customer_id": customer_id,
        "segment_id": segment_id,
        "segment_name": segment_name,
        "characteristics": chars
    }

def get_recommendation(customer_id: int):
    # Rule based Next Best Offer using segment and credit risk
    segment_data = get_segmentation(customer_id)
    risk_data = get_credit_risk(customer_id)
    
    if not segment_data or not risk_data: return None
    
    segment = segment_data['segment_name']
    risk = risk_data['risk_category']
    chars = segment_data['characteristics']
    
    if risk == "HIGH":
        product = "Debt Consolidation Consultation"
        reasons = ["High credit risk", "Potential financial distress"]
        conf = 0.85
    elif segment == "Low-Engagement Savers" and risk == "LOW":
        product = "Premium Rewards Credit Card"
        reasons = ["Low current utilization", "Excellent credit profile", "Opportunity to increase engagement"]
        conf = 0.90
    elif segment == "High-Utilization Borrowers" and risk == "LOW":
        product = "Personal Loan (Lower Interest)"
        reasons = ["High revolving utilization", "Good payment behavior", "Can reduce interest burden"]
        conf = 0.75
    elif "Struggling" in segment:
        product = "Repayment Assistance Plan"
        reasons = ["Historical payment delays observed"]
        conf = 0.80
    else:
        product = "Savings / Investment Account"
        reasons = ["Balanced utilization", "Stable behavior"]
        conf = 0.65
        
    return {
        "customer_id": customer_id,
        "recommended_product": product,
        "confidence": conf,
        "reasons": reasons
    }

def get_model_metrics():
    artifacts = load_artifacts()
    return {
        "credit_risk": artifacts['credit_metrics']
    }
