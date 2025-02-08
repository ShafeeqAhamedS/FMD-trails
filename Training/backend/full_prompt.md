### Prompt

**Objective:**  
We need to create a production-ready FastAPI service for a machine learning (ML) model. The service should provide endpoints for the following functionalities:

1. **Health Check Endpoint:**  
   An endpoint `/health` to verify the API is running and the model is loaded.

2. **Prediction Endpoint:**  
   An endpoint `/predict` that accepts a JSON payload with an "inputs" object.  
   Each value in the "inputs" dictionary should accept strings, floats, or integers, as specified in the attached JSON code block.  
   - The prediction logic should convert the inputs into appropriate data types before invoking the model.
   - The prediction response should include the output, which could be a result of processing, transformed data, or model inference.  

3. **Metrics Endpoint:**  
   The `/metrics` endpoint should return model evaluation metrics (accuracy, confusion matrix, R1 score, etc.) only if they are available.  
   If no metrics are available, the API should return "No metrics available".

4. **Error Handling and Logging:**  
   - Proper error handling using `try/except` blocks.
   - Log all errors appropriately.
   - Ensure that the API returns clear error messages if the model is not loaded or if input conversion fails.

5. **Startup and Shutdown Events:**  
   The model should be loaded from a pickle file during startup. If the model fails to load, return a service unavailable status.

6. **Dynamic Metrics:**  
   If evaluation metrics such as accuracy and classification matrix are provided in the JSON block, include them in the response. If no metrics are provided, indicate the lack of metrics.



#### 1. Dataset Preview

         Position  Level  Salary
 Business Analyst      1   45000
Junior Consultant      2   50000
              ...    ...     ...
          C-level      9  500000
              CEO     10 1000000

#### 2. Code Blocks JSON

[
    {
        "source": [
            "import pandas as pd\n",
            "d=pd.read_csv(\"./Position_Salaries.csv\")"
        ],
        "outputs": []
    },
    {
        "source": [
            "d.head()\n",
            "\n"
        ],
        "outputs": [
            [
                "            Position  Level  Salary\n",
                "0   Business Analyst      1   45000\n",
                "1  Junior Consultant      2   50000\n",
                "2  Senior Consultant      3   60000\n",
                "3            Manager      4   80000\n",
                "4    Country Manager      5  110000"
            ]
        ]
    },
    {
        "source": [
            "d.info()"
        ],
        "outputs": []
    },
    {
        "source": [
            "d.isnull().sum()"
        ],
        "outputs": [
            [
                "Position    0\n",
                "Level       0\n",
                "Salary      0\n",
                "dtype: int64"
            ]
        ]
    },
    {
        "source": [
            "from sklearn.preprocessing import LabelEncoder\n",
            "l=LabelEncoder()\n",
            "d[\"Position\"] = l.fit_transform(d[\"Position\"])"
        ],
        "outputs": []
    },
    {
        "source": [
            "d.head()"
        ],
        "outputs": [
            [
                "   Position  Level  Salary\n",
                "0         0      1   45000\n",
                "1         4      2   50000\n",
                "2         8      3   60000\n",
                "3         5      4   80000\n",
                "4         3      5  110000"
            ]
        ]
    },
    {
        "source": [
            "x = d[[\"Position\",\"Level\"]]\n",
            "y = d[\"Salary\"]"
        ],
        "outputs": []
    },
    {
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)"
        ],
        "outputs": []
    },
    {
        "source": [
            "from sklearn.tree import DecisionTreeRegressor\n",
            "dt = DecisionTreeRegressor()"
        ],
        "outputs": []
    },
    {
        "source": [
            "dt.fit(x_train,y_train)\n",
            "y_pred = dt.predict(x_test)"
        ],
        "outputs": []
    },
    {
        "source": [
            "from sklearn import metrics\n",
            "mse = metrics.mean_squared_error(y_test,y_pred)\n",
            "mse"
        ],
        "outputs": [
            [
                "462500000.0"
            ]
        ]
    },
    {
        "source": [
            "r2 = metrics.r2_score(y_test,y_pred)\n",
            "r2"
        ],
        "outputs": [
            [
                "0.48611111111111116"
            ]
        ]
    },
    {
        "source": [
            "pred = dt.predict([[2,2]])\n",
            "pred"
        ],
        "outputs": [
            [
                "array([45000.])"
            ]
        ]
    },
    {
        "source": [
            "# saving the model to the local file system\n",
            "import pickle\n",
            "filename = 'finalized_model.pickle'\n",
            "pickle.dump(dt, open(filename, 'wb'))"
        ],
        "outputs": []
    },
    {
        "source": [],
        "outputs": []
    }
] 

 2025-02-08 19:47:13,160 - INFO - Dataset loaded successfully.
2025-02-08 19:47:13,161 - INFO - Encoded 'Position' column successfully.
2025-02-08 19:47:13,286 - INFO - Model loaded successfully.
2025-02-08 19:47:13,288 - INFO - Prediction made successfully: [80000.]
2025-02-08 19:47:13,290 - INFO - MAE: 17500.0
2025-02-08 19:47:13,290 - INFO - MSE: 462500000.0
2025-02-08 19:47:13,290 - INFO - RMSE: 21505.813167606568
2025-02-08 19:47:13,291 - INFO - R²: 0.48611111111111116
2025-02-08 19:47:13,291 - INFO - MAPE: 18.636363636363633
2025-02-08 19:47:13,291 - INFO - Sample Prediction: [80000.]
2025-02-08 19:47:13,291 - INFO - Model Evaluation - MSE: 462500000.0, R2: 0.48611111111111116
#### 3. Template Code

### `main.py` Template with Generalized Description:

```python
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
    title="TITLE",  # Change to appropriate Title based on JSON block
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
    # Example: { "inputs": { "feature1": 0.5, "feature2": "sample text", "feature3": 10 } }
    inputs: Dict[str, Any] = Field(..., description="A dictionary of input values (int, float, or string).")
    
    class Config:
        extra = Extra.forbid  # Forbid unexpected fields at the top level

# -------------------------- 
# Global Variables & Metrics 
# --------------------------

# Global model reference (loaded during startup)
model = None

# -------------------------
# Utility: Model Prediction
# -------------------------
def model_predict(model, inputs: Dict[str, Any]) -> Any:
    """
    Prediction function. Replace this with actual model inference logic.
    
    Example, if using a scikit-learn model:
        processed_inputs = preprocess(inputs)
        prediction = model.predict(processed_inputs)
    """
    # Here, simply return a dummy result for illustration
    return [f"Processed: {input_value}" for input_value in inputs.values()]

# ---------------------------
# Startup and Shutdown Events
# ---------------------------
@app.on_event("startup")
def load_model():
    """
    Load the model during startup from a pickle file.
    """
    global model
    model_path = "model.pkl"  # update with the model file path based on given JSON block

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Optionally, load or set evaluation metrics if available.
        # For example, Get attributes from the Code and Output Block, if not given, set as None.
        if not hasattr(model, "accuracy"):
            model.accuracy = None  # Example value; replace with your actual metric
        if not hasattr(model, "classification_matrix"):
            model.classification_matrix = {
                "true_positive": None,
                "false_positive": None,
                "true_negative": None,
                "false_negative": None
            }
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
    return {"status": "ok"}


@app.post("/predict", tags=["Prediction"])
async def predict(payload: PredictionRequest):
    """
    Prediction endpoint.
    Expects a JSON payload with a dictionary of inputs, which may include strings, integers, or floats.
    Returns model predictions or processed results.
    
    Example Input: 
    {
        "inputs": {
            "feature1": 0.5,
            "feature2": "sample text",
            "feature3": 10
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
    Returns model evaluation metrics (accuracy, classification matrix) only if available.
    """
    # Try to retrieve model evaluation metrics if they exist.
    model_accuracy = getattr(model, 'accuracy', None) if model else None
    classification_matrix = getattr(model, 'classification_matrix', None) if model else None
    model_r1_score = getattr(model, 'r1_score', None) if model else None

    # If metrics are available, return them, otherwise, indicate no metrics.
    if model_accuracy or classification_matrix or model_r1_score:
        return {
            "accuracy": model_accuracy,
            "classification_matrix": classification_matrix,
            "r1_score": model_r1_score
        }
    else:
        return {"message": "No metrics available"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```