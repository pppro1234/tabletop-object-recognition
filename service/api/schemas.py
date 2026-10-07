from typing import List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    image_path: str


class DetectionResponse(BaseModel):
    track_id: int
    label: str
    score: float
    centroid: List[float]
    angle_deg: float


class PredictResponse(BaseModel):
    detections: List[DetectionResponse]
