
# main.py

import time
import pickle
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, Extra
import uvicorn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Setup logger
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Student Score Prediction API",  # Change to appropriate Title based on JSON block
    description="FastAPI service for serving predictions and evaluation metrics for the ML model.",
    version="1.0.0"
)

# Allow CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Pydantic Models for Validation
# -------------------------------

class PredictionRequest(BaseModel):
    # Accepts an arbitrary number of inputs in key-value form.
    # Example: { "inputs": { "Hours": 5.0} }
    inputs: Dict[str, Any] = Field(..., description="A dictionary of input values (int, float, or string).")
    
    class Config:
        extra = Extra.forbid  # Forbid unexpected fields at the top level

# -------------------------- 
# Global Variables & Metrics 
# --------------------------

# Global model reference (loaded during startup)
model = None
mae = None
mse = None
rmse = None
r2 = None
mape = None

# -------------------------
# Utility: Model Prediction
# -------------------------
def model_predict(model: LinearRegression, inputs: Dict[str, Any]) -> Any:
    """
    Prediction function. Replace this with actual model inference logic.
    """
    try:
        hours = float(inputs.get("Hours"))
        prediction = model.predict(np.array([[hours]]))
        return prediction.tolist()
    except (ValueError, TypeError) as e:
        logger.error(f"Error during input conversion or prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input format. Please provide a valid number for 'Hours'."
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction."
        )

# ---------------------------
# Startup and Shutdown Events
# ---------------------------
@app.on_event("startup")
def load_model():
    """
    Load the model during startup from a pickle file.
    """
    global model, mae, mse, rmse, r2, mape
    model_path = "model.pkl"  # update with the model file path based on given JSON block

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        
        mae = 4.691397441397446
        mse = 25.463280738222593
        rmse = 5.046115410711748
        r2 = 0.9555700801388128
        mape = 11.581971484864278
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        model = None


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down the ML API server.")


# -----------------------
# Exception Handlers
# -----------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An internal error occurred. Please try again later."},
    )


# -------------------
# API Route Handlers
# -------------------

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint. Returns a simple status message to confirm the API is running and the model is loaded.
    """
    if model is not None:
         return {"status": "ok"}
    else:
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later."
        )


@app.post("/predict", tags=["Prediction"])
async def predict(payload: PredictionRequest):
    """
    Prediction endpoint.
    Expects a JSON payload with a dictionary of inputs, which may include strings, integers, or floats.
    Returns model predictions or processed results.
    
    Example Input: 
    {
        "inputs": {
            "Hours": 5.0
        }
    }
    """
    global model

    if model is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later."
        )

    try:
        # Call the model_predict function which should contain your actual inference logic.
        predictions = model_predict(model, payload.inputs)
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction."
        )

    return {"predictions": predictions}


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """
    Metrics endpoint.
    Returns model evaluation metrics (MAE, MSE, RMSE, R², MAPE) only if available.
    """
    global mae, mse, rmse, r2, mape
    
    if all(v is not None for v in [mae, mse, rmse, r2, mape]):
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2,
            "MAPE": mape
        }
    else:
        return {"message": "No metrics available"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
