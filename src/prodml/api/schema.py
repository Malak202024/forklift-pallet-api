from pydantic import BaseModel

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

class PredictResponse(BaseModel):
    status: str
    detections: list[DetectionItem]