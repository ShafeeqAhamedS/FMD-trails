### Prompt

### Prompt

**Objective:**  
You are an AI code generator. Your task is to generate a Python script (`model.py`) that loads a saved machine learning model, preprocesses input data, runs inference, and evaluates the model's performance. The script should be production-ready, modular, and follow best practices for error handling, logging, and code structure.

**Inputs Provided:**  
1. **Template Python File**: A skeleton script with placeholders for key functionalities.  
2. **Dataset Preview**: A preview of the dataset (column names and sample rows).  
3. **Code Blocks JSON**: JSON containing Python code blocks used during training (e.g., preprocessing, model training, evaluation).  

**Requirements:**  
The script must:  
1. **Load the Dataset**: Load the dataset from a CSV file and validate its structure. Always Add './' while reading file
2. **Preprocess Data**: Apply necessary preprocessing steps (e.g., encoding, scaling, feature engineering) based on the training pipeline.  
3. **Load the Model**: Load a pre-trained model from a file (e.g., `.pkl`, `.joblib`).  Always Add './' while loading model
4. **Run Inference**: Select a sample row from the dataset and generate predictions using the loaded model.  
5. **Evaluate the Model**: Compute evaluation metrics (e.g., accuracy, F1-score for classification; MSE, R² for regression).  
6. **Error Handling**: Include robust exception handling for missing data, incorrect data types, or model loading issues.  
7. **Logging**: Use logging to track the script's execution and report errors or important events.  
8. **Modularity**: Break the code into reusable functions (e.g., `load_dataset`, `preprocess_data`, `load_model`, `model_inference`, `model_evaluation`).  
9. **Generalization**: Avoid hardcoding column names, file paths, or model-specific details. Use variables and configurations that can be easily modified.  

**Expected Output:**  
A well-structured Python script (`model.py`) that implements the above functionality. The script should be clear, well-commented, and ready for production use.

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

 None#### 3. Template Code

Template: model.py
This script:
✅ Loads the dataset preview and extracts one row
✅ Detects preprocessing steps (scaling, encoding, feature extraction)
✅ Loads the saved model and runs inference
✅ Evaluates model performance
Change the file names, paths, and model loading based on the given code blocks.  
Change the preprocessing steps based on the data types and code blocks.  
Change the evaluation metrics based on the target variable type.  
Do not hardcode column names or model paths.  
Do not include training code or model saving.  

```py
# Import necessary libraries (pandas, numpy, scikit-learn, etc, based on given code blocks)
import joblib  # To load the trained model
import logging # To log messages
from sklearn.model_selection import train_test_split # To split the dataset into training and testing sets

# Ignore sklearn warnings
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load dataset (Replace 'dataset.csv' with actual dataset file if required)
def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        exit()

def preprocess_data(df):
    """Apply preprocessing steps like scaling and encoding based on the data types and code blocks, do only the necessary steps as per the given code blocks."""
    try:
        # Encoding
        logging.info("encoded successfully.")
        
        # Scaling
        logging.info("Features scaled successfully.")
        
        # Feature Engineering
        logging.info("Feature engineering done.")
        
        
        return df
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        exit()

def load_model(model_path):
    """Load the pre-trained model from a file."""
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

def model_inference(model, sample_row):
    """Make prediction using the trained model."""
    try:
        prediction = model.predict(sample_row)
        logging.info(f"Prediction made successfully: {prediction}")
        return prediction
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        exit()

def model_evaluation(model, X_test, y_test):
    """Evaluate model performance using MSE and R2."""
    try:
        # Predicting the target variable
        y_pred = model.predict(X_test)
        
        if y.dtype == "object":  # Classification case (if target is categorical) (Use all classification metrics)
            # Use all 5 classification metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(y, y_pred, average='weighted')
            cm = confusion_matrix(y, y_pred)
            
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1-Score: {f1}")
            logging.info(f"Confusion Matrix:\n{cm}")
        else:  # Regression case
            # Calculate regression metrics, use all regression metrics
            # Use all 5 regression metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / y)) * 100

            # Log and print the regression evaluation results
            logging.info(f"MAE: {mae}")
            logging.info(f"MSE: {mse}")
            logging.info(f"RMSE: {rmse}")
            logging.info(f"R²: {r2}")
            logging.info(f"MAPE: {mape}")
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        exit()

def main():
    # Load dataset (Replace 'dataset.csv' with actual dataset file if required)
    df = load_dataset("./dataset.csv") # Always load dataset from the same directory as the script ./

    # Detect numerical and categorical columns for preprocessing based on given code blocks
    # Preprocess data
    df = preprocess_data(df)
    
    # Prepare features and target variable
    # Detect target column and features based on given code blocks
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Load trained model (Replace 'model.pkl' with actual trained model file based on given code blocks)
    model = load_model("model.pkl")  # Always load model from the same directory as the script ./
    
    # Run model inference on a sample row
    sample_row = X_test.sample(1)  # Select one row for prediction based on given code blocks
    prediction = model_inference(model, sample_row)
    
    # Evaluate the model (All evaluation metrics based on the target variable type)
    model_evaluation(model, X_test, y_test)
    
    # Print results
    logging.info(f"Sample Prediction: {prediction}")
    logging.info(f"Model Evaluation - MSE: {mse}, R2: {r2}")

if __name__ == "__main__":
    main()
```