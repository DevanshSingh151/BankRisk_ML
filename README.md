# BankRisk360: Intelligent Banking Decision & Risk Analytics Platform

## Project Overview
BankRisk360 is a fully functional Data Science and Decision Management prototype designed for the modern banking environment. Instead of treating Machine Learning purely as an academic exercise, BankRisk360 implements the **Predict → Explain → Recommend** framework. 

For every automated ML prediction, the system explains *why* the model made that decision and outputs an actionable *banking recommendation* to support human review and intervention.

## Problem Statement
A customer may have limited traditional credit history but demonstrate stable income and responsible financial behavior through other available signals. Traditional rules-based banking systems might automatically reject these thin-file customers or fail to notice early signs of delinquency in existing customers. 

BankRisk360 attempts to extract useful behavioral signals from available transactional data to support:
- Credit assessment
- Early risk detection
- Customer understanding
- Product recommendation

**Important Note:** The system does not replace human banking decisions. It provides actionable decision-support directly alongside its probability outputs.

## Dataset Sources
- **Dataset:** Default of Credit Card Clients Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Description:** Contains 30,000 anonymized records with features including demographic factors, historical payment statuses, bill amounts, and previous payments over 6 months.

## Tech Stack
- **Backend**: Python, FastAPI, Uvicorn, Pydantic, SQLAlchemy
- **Data Science / ML**: Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, SHAP, Joblib
- **Frontend**: Streamlit, Plotly (for interactive visualizations)
- **Database**: SQLite (managed via SQLAlchemy for easy migration to PostgreSQL)


## Architecture
```text
                    PUBLIC DATASET
                         |
                DATA PREPROCESSING & 
                 FEATURE ENGINEERING
                         |
          +--------------+--------------+
          |              |              |
    CREDIT RISK     DELINQUENCY     CUSTOMER
      MODEL           MODEL       SEGMENTATION
          |              |              |
          +--------------+--------------+
                         |
                 DECISION ENGINE
                         |
              +----------+----------+
              |                     |
        EXPLAINABILITY       NEXT-BEST-OFFER
              |                     |
              +----------+----------+
                         |
                    FASTAPI BACKEND
                         |
                   STREAMLIT FRONTEND
```

## ML Methodology
### Feature Engineering
We engineered several banking-specific behavioral features from the raw transactional data, including:
- **Average Credit Utilization:** Sum of billed amounts divided by credit limit.
- **Payment Ratio:** Total payments divided by total billed amounts.
- **Max Delay:** Maximum months of delay across the historical period.

### Why XGBoost?
We selected **XGBoost** for both our Credit Risk and Early Warning models due to its strong performance on tabular data, its robustness to outliers, its native handling of missing values, and its excellent integration with SHAP for individual-level explainability.

### Class Imbalance Handling
The dataset target is naturally imbalanced. To handle this without introducing data leakage, we applied **SMOTE (Synthetic Minority Over-sampling Technique)** *strictly on the training partition* after the train/test split. Evaluation metrics are recorded purely against the real-world distribution in the holdout test set.

### Explainability with SHAP
We use TreeExplainer from the SHAP library. This allows BankRisk360 to break down the model's complex prediction into discrete, understandable feature contributions (e.g. "+0.14 due to High Credit Utilization"). 

### Customer Segmentation
K-Means clustering was applied to continuous behavioral features (utilization, payment ratio, age) after standardization. The centers were analyzed to dynamically label the clusters (e.g. "Struggling Borrowers", "Low-Engagement Savers") to support the Next-Best-Offer engine.

## API Documentation
The FastAPI backend exposes the following endpoints:
- `GET /api/health` - API health check
- `GET /api/customer/{id}` - Retrieve raw customer profile
- `POST /api/predict/credit-risk` - Returns risk score and category
- `GET /api/predict/credit-risk/explain/{id}` - Returns SHAP top drivers and reducers
- `POST /api/predict/delinquency` - Returns short-term delinquency risk
- `GET /api/customer/{id}/segment` - Returns assigned behavioral cluster
- `GET /api/customer/{id}/recommendation` - Returns Next-Best-Offer

## Project Structure
- `data/` - Raw downloaded dataset and engineered dataset
- `models/` - Saved Joblib artifacts (XGBoost, Scalers, KMeans)
- `backend/` - FastAPI application, Routers, Pydantic schemas, ML Service layer
- `frontend/` - Streamlit application, Dashboard UI, Customer 360 View
- `scripts/` - Automated pipeline scripts (`download_data.py`, `train_models.py`)

## Installation & How to Run
The application runs entirely locally.

### 1. Create a Virtual Environment and Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Application Using the Startup Script
We have provided a unified runner that will automatically download the dataset, train the models, and launch both backend and frontend servers:
```bash
python run.py
```
*Wait for the console to indicate that both Uvicorn and Streamlit have started. The Streamlit dashboard will typically open automatically at http://localhost:8501.*

### Manual Startup (Alternative)
If you prefer to run services manually in separate terminals:
**Terminal 1:**
```bash
python scripts/download_data.py
python scripts/train_models.py
uvicorn backend.main:app --reload
```
**Terminal 2:**
```bash
streamlit run frontend/app.py
```

## Limitations & Future Improvements
- **Data Limitations:** The temporal nature of "early warning" is simulated cross-sectionally since we only have 6 months of historical state for each user. A true time-series dataset would allow more robust horizon modeling.
- **Model Deployment:** The current setup uses a SQLite database and local model artifacts. A production version would migrate to PostgreSQL and utilize an MLflow Model Registry for versioning.
- **Advanced Recommendations:** The Next-Best-Offer engine currently acts as an expert-system rules layer on top of ML segments. With actual product holding data, this could be upgraded to a collaborative filtering or sequence-prediction model.
