import subprocess
import sys

packages = [
    "fastapi==0.111.0",
    "uvicorn==0.30.1",
    "sqlalchemy==2.0.30",
    "pydantic==2.7.4",
    "pandas==2.2.2",
    "numpy==1.26.4",
    "scikit-learn==1.5.0",
    "xgboost==2.0.3",
    "imbalanced-learn==0.12.3",
    "shap==0.45.1",
    "plotly==5.22.0",
    "pytest==8.2.2",
    "requests==2.32.3",
    "python-dotenv==1.0.1",
    "joblib==1.4.2",
    "scipy==1.13.1",
    "matplotlib==3.9.0",
    "streamlit==1.36.0"
]

for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--no-cache-dir"])
    print(f"Finished {pkg}")

print("All packages installed.")
