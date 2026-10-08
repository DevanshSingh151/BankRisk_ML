from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class CustomerPredictionRequest(BaseModel):
    customer_id: int

class PredictionResponse(BaseModel):
    customer_id: int
    probability: float
    risk_score: int
    risk_category: str
    action: str

class ShapFeature(BaseModel):
    feature: str
    value: float
    contribution: float

class ExplainabilityResponse(BaseModel):
    customer_id: int
    top_risk_drivers: List[ShapFeature]
    top_risk_reducers: List[ShapFeature]
    base_value: float

class SegmentationResponse(BaseModel):
    customer_id: int
    segment_id: int
    segment_name: str
    characteristics: Dict[str, float]

class RecommendationResponse(BaseModel):
    customer_id: int
    recommended_product: str
    confidence: float
    reasons: List[str]

class ModelMetricsResponse(BaseModel):
    credit_risk: Dict[str, Any]
    delinquency: Optional[Dict[str, Any]] = None
