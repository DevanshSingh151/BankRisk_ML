import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def feature_engineering(df):
    df_engineered = df.copy()
    
    # 1. Credit Utilization
    # Sum of bills over 6 months / (6 * limit_bal)
    bill_cols = ['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']
    df_engineered['avg_utilization'] = df_engineered[bill_cols].mean(axis=1) / df_engineered['limit_bal']
    df_engineered['avg_utilization'] = df_engineered['avg_utilization'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 2. Payment Ratio
    # Sum of payments / Sum of bills
    pay_cols = ['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']
    total_bill = df_engineered[bill_cols].sum(axis=1).replace(0, 1) # avoid division by zero
    total_pay = df_engineered[pay_cols].sum(axis=1)
    df_engineered['payment_ratio'] = total_pay / total_bill
    df_engineered['payment_ratio'] = df_engineered['payment_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Delinquency History (Max delay in past 6 months)
    delay_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    # Sometimes pay_0 is actually named pay_1 in some versions of dataset. OpenML uses pay_0
    if 'pay_0' not in df_engineered.columns and 'pay_1' in df_engineered.columns:
        df_engineered.rename(columns={'pay_1': 'pay_0'}, inplace=True)
        delay_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
        
    df_engineered['max_delay'] = df_engineered[delay_cols].max(axis=1)
    df_engineered['max_delay'] = df_engineered['max_delay'].apply(lambda x: x if x > 0 else 0)
    
    # Cast target and categorical variables to numeric
    if 'default_payment_next_month' in df_engineered.columns:
        df_engineered['default_payment_next_month'] = pd.to_numeric(df_engineered['default_payment_next_month'])
    for col in ['sex', 'education', 'marriage'] + delay_cols:
        df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce').fillna(0)
        
    return df_engineered

def train_credit_risk_model(df, models_dir):
    print("--- Training Credit Risk Model ---")
    features = [
        'limit_bal', 'sex', 'education', 'marriage', 'age',
        'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
        'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
        'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
        'avg_utilization', 'payment_ratio', 'max_delay'
    ]
    target = 'default_payment_next_month'
    
    X = df[features]
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle Class Imbalance with SMOTE on training data ONLY
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train_res)
    
    # Evaluate
    y_pred = xgb_model.predict(X_test_scaled)
    y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob)
    }
    print(f"Metrics: {metrics}")
    
    # Save artifacts
    joblib.dump(scaler, os.path.join(models_dir, 'credit_scaler.joblib'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'credit_xgb.joblib'))
    joblib.dump(features, os.path.join(models_dir, 'credit_features.joblib'))
    joblib.dump(metrics, os.path.join(models_dir, 'credit_metrics.joblib'))
    
def train_delinquency_model(df, models_dir):
    print("--- Training Early Warning Model ---")
    # Predict if they are currently late (pay_0 >= 2) using only historical data (month 2-6)
    # This simulates "predicting next month's delinquency"
    
    df_dw = df.copy()
    df_dw['target_delinquency'] = (pd.to_numeric(df_dw['pay_0']) >= 2).astype(int)
    
    features = [
        'limit_bal', 'sex', 'education', 'marriage', 'age',
        'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
        'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
        'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'
    ]
    
    X = df_dw[features]
    y = df_dw['target_delinquency']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    # Use scale_pos_weight since it's imbalanced and we won't use SMOTE here for variety
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Save
    joblib.dump(scaler, os.path.join(models_dir, 'delinquency_scaler.joblib'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'delinquency_xgb.joblib'))
    joblib.dump(features, os.path.join(models_dir, 'delinquency_features.joblib'))

def train_segmentation_model(df, models_dir):
    print("--- Training Customer Segmentation Model ---")
    features = ['limit_bal', 'age', 'avg_utilization', 'payment_ratio', 'max_delay']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # PCA for visualization (2 components)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    # Save
    joblib.dump(scaler, os.path.join(models_dir, 'segmentation_scaler.joblib'))
    joblib.dump(kmeans, os.path.join(models_dir, 'kmeans.joblib'))
    joblib.dump(pca, os.path.join(models_dir, 'pca.joblib'))
    joblib.dump(features, os.path.join(models_dir, 'segmentation_features.joblib'))

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'raw_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run download_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Preprocess & Feature Engineer
    df_engineered = feature_engineering(df)
    df_engineered.to_csv(os.path.join(base_dir, 'data', 'processed_data.csv'), index=False)
    
    train_credit_risk_model(df_engineered, models_dir)
    train_delinquency_model(df_engineered, models_dir)
    train_segmentation_model(df_engineered, models_dir)
    print("All models trained and saved successfully.")

if __name__ == '__main__':
    main()
