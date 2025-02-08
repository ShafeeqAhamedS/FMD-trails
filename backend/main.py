
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

# Setup logger
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Salary Prediction API",  # Change to appropriate Title based on JSON block
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
    # Example: { "inputs": { "Position": 0, "Level": 1} }
    inputs: Dict[str, Any] = Field(..., description="A dictionary of input values (int, float, or string).")
    
    class Config:
        extra = Extra.forbid  # Forbid unexpected fields at the top level

# -------------------------- 
# Global Variables & Metrics 
# --------------------------

# Global model reference (loaded during startup)
model = None
mse = None
r2 = None

# -------------------------
# Utility: Model Prediction
# -------------------------
def model_predict(model, inputs: Dict[str, Any]) -> Any:
    """
    Prediction function.
    
    Accepts a dictionary of inputs with keys "Position" and "Level", converts them to the correct types,
    and returns the model's prediction.
    """
    try:
        position = int(inputs["Position"])
        level = int(inputs["Level"])
        
        # Make sure the input is a 2D array
        input_data = [[position, level]]

        prediction = model.predict(input_data)
        return prediction.tolist()  # Return as list for JSON serialization
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid input: {e}")
        raise ValueError("Invalid input format. Please provide 'Position' and 'Level' as integers.")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise Exception("An error occurred during prediction.")

# ---------------------------
# Startup and Shutdown Events
# ---------------------------
@app.on_event("startup")
def load_model_event():
    """
    Load the model during startup from a pickle file.
    """
    global model
    global mse
    global r2
    model_path = "./finalized_model.pickle"  # update with the model file path based on given JSON block

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")

        # Set evaluation metrics if available.
        mse = 462500000.0
        r2 = 0.48611111111111116

    except FileNotFoundError as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        model = None
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model file not found."
        )
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        model = None
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to load model."
        )


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
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later."
        )
    return {"status": "ok"}


@app.post("/predict", tags=["Prediction"])
async def predict(payload: PredictionRequest):
    """
    Prediction endpoint.
    Expects a JSON payload with a dictionary of inputs containing "Position" and "Level" as integers.
    Returns model predictions.
    
    Example Input: 
    {
        "inputs": {
            "Position": 2,
            "Level": 2
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
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
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
    Returns model evaluation metrics (MSE and R2) if available.
    """
    global mse
    global r2

    if mse is not None and r2 is not None:
        return {
            "mse": mse,
            "r2": r2
        }
    else:
        return {"message": "No metrics available"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
