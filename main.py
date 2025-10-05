"""Minimal FastAPI bootstrap that wires modular routes and startup load."""
from fastapi import FastAPI
import os
from app.routes import router, startup_load
from app import config

app = FastAPI(title="Hotel Cancellation Prediction API", version=config.APP_VERSION)

@app.on_event("startup")
async def _load():
    startup_load()

@app.get("/", response_model=dict)
async def root():
    return {"message": "Hotel Cancellation Prediction API", "version": config.APP_VERSION, "endpoints": {"health": "/health", "predict": "/predict", "docs": "/docs"}}

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
