"""Pydantic request/response models."""
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class BookingFeatures(BaseModel):
    """Public prediction payload schema.

    Many training-time features are internal or engineered; to keep the public
    contract lightweight we make several fields optional with neutral defaults.
    This enables a *minimal* JSON payload such as:
        {"lead_time": 30, "arrival_month": 7, "adults": 2, "children": 0, "adr": 120.0}
    The `_prepare` function supplements / engineers the remaining columns.
    """

    lead_time: int = Field(..., ge=0)
    arrival_month: int = Field(..., ge=1, le=12)
    # Stay details (optional, defaulting to a short weekday stay)
    stays_weekend_nights: int | None = Field(0, ge=0)
    stays_week_nights: int | None = Field(1, ge=0)
    # Guest composition
    adults: int = Field(..., ge=1)
    children: int | None = Field(0, ge=0)
    # Historical / behavioral signals
    is_repeated_guest: int | None = Field(0, ge=0, le=1)
    previous_cancellations: int | None = Field(0, ge=0)
    booking_changes: int | None = Field(0, ge=0)
    # Pricing
    adr: float = Field(..., ge=0, description="Average daily rate (numeric, required)")
    # Amenities / request counts
    required_car_parking_spaces: int | None = Field(0, ge=0)
    total_of_special_requests: int | None = Field(0, ge=0)

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "summary": "Minimal",
                "value": {
                    "lead_time": 30,
                    "arrival_month": 7,
                    "adults": 2,
                    "children": 0,
                    "adr": 120.0
                }
            },
            {
                "summary": "Extended",
                "value": {
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
            }
        ]
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
