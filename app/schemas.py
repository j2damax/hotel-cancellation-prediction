"""Pydantic request/response models."""
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class BookingFeatures(BaseModel):
    lead_time: int = Field(..., ge=0)
    arrival_month: int = Field(..., ge=1, le=12)
    stays_weekend_nights: int = Field(..., ge=0)
    stays_week_nights: int = Field(..., ge=0)
    adults: int = Field(..., ge=1)
    children: int = Field(..., ge=0)
    is_repeated_guest: int = Field(..., ge=0, le=1)
    previous_cancellations: int = Field(..., ge=0)
    booking_changes: int = Field(..., ge=0)
    adr: float = Field(..., ge=0)
    required_car_parking_spaces: int = Field(..., ge=0)
    total_of_special_requests: int = Field(..., ge=0)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "lead_time": 120,
            "arrival_month": 7,
            "stays_weekend_nights": 2,
            "stays_week_nights": 3,
            "adults": 2,
            "children": 1,
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "booking_changes": 1,
            "adr": 95.5,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 2
        }
    })


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str | None = None
    applied_threshold: float | None = None
    threshold_source: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    decision_threshold: Optional[float] = None
