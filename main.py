"""Minimal FastAPI bootstrap using lifespan for startup model load."""
from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
from app.routes import router, startup_load
from app import config


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[override]
    # Startup
    try:
        startup_load()
    except Exception as e:
        # Allow continuation if configured; error surfaces via /health
        if not config.ALLOW_START_WITHOUT_MODEL:
            raise
        print(f"Startup load warning (continuing due to ALLOW_START_WITHOUT_MODEL): {e}")
    yield
    # Shutdown (no-op for now; placeholder for future resource cleanup)


app = FastAPI(title="Hotel Cancellation Prediction API", version=config.APP_VERSION, lifespan=lifespan)

@app.get("/", response_model=dict)
async def root():
    return {"message": "Hotel Cancellation Prediction API", "version": config.APP_VERSION, "endpoints": {"health": "/health", "predict": "/predict", "docs": "/docs"}}

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
