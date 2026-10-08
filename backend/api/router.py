from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from backend.schemas import schemas
from backend.services import ml_service

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "message": "BankRisk360 API is running"}

@router.get("/info")
def get_info():
    info = ml_service.get_all_customers_info()
    return info

@router.get("/customer/{customer_id}", response_model=Dict[str, Any])
def get_customer(customer_id: int):
    data = ml_service.get_customer_data(customer_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return data.to_dict()

@router.post("/predict/credit-risk", response_model=schemas.PredictionResponse)
def predict_credit_risk(request: schemas.CustomerPredictionRequest):
    result = ml_service.get_credit_risk(request.customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result

@router.get("/predict/credit-risk/explain/{customer_id}", response_model=schemas.ExplainabilityResponse)
def explain_credit_risk(customer_id: int):
    result = ml_service.get_credit_explanation(customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result

@router.post("/predict/delinquency", response_model=schemas.PredictionResponse)
def predict_delinquency(request: schemas.CustomerPredictionRequest):
    result = ml_service.get_delinquency_risk(request.customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result

@router.get("/customer/{customer_id}/segment", response_model=schemas.SegmentationResponse)
def get_segment(customer_id: int):
    result = ml_service.get_segmentation(customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result

@router.get("/customer/{customer_id}/recommendation", response_model=schemas.RecommendationResponse)
def get_recommendation(customer_id: int):
    result = ml_service.get_recommendation(customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result

@router.get("/model/metrics", response_model=schemas.ModelMetricsResponse)
def get_metrics():
    return ml_service.get_model_metrics()
