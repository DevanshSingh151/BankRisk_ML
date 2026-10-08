import subprocess
import os
import sys
import time
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for dataset
    data_path = os.path.join(base_dir, 'data', 'raw_data.csv')
    if not os.path.exists(data_path):
        print("Downloading dataset...")
        subprocess.run([sys.executable, os.path.join(base_dir, 'scripts', 'download_data.py')])
        
    # Check for models
    model_path = os.path.join(base_dir, 'models', 'credit_xgb.joblib')
    if not os.path.exists(model_path):
        print("Training models... (This may take a minute)")
        subprocess.run([sys.executable, os.path.join(base_dir, 'scripts', 'train_models.py')])
        
    # Start Backend
    print("Starting FastAPI Backend on port 8000...")
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000"
    ])
    
    # Wait a bit for backend to start
    time.sleep(5)
    
    # Start Frontend
    print("Starting Streamlit Frontend...")
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", os.path.join(base_dir, 'frontend', 'app.py'), "--server.port", "8501"
    ])
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("Shutting down processes...")
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    run()
