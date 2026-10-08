import pytest
from fastapi.testclient import TestClient
import os
import sys

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "BankRisk360 API is running"}

def test_get_customer_not_found():
    # Large ID that doesn't exist
    response = client.get("/api/customer/999999")
    assert response.status_code == 404
