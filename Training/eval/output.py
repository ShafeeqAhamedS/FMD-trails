
# Import necessary libraries (pandas, numpy, scikit-learn, etc, based on given code blocks)
import pandas as pd
import numpy as np
import joblib  # To load the trained model
import logging # To log messages
from sklearn.model_selection import train_test_split # To split the dataset into training and testing sets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# For Classification Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Ignore sklearn warnings
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset (Replace 'dataset.csv' with actual dataset file if required)
def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        return df
    except FileNotFoundError as e:
        logging.error(f"Error loading dataset: File not found at {file_path}")
        exit()
    except pd.errors.EmptyDataError as e:
        logging.error(f"Error loading dataset: The file is empty.")
        exit()
    except pd.errors.ParserError as e:
        logging.error(f"Error loading dataset: Parsing error. Check the file format.")
        exit()
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        exit()

def preprocess_data(df):
    """Apply preprocessing steps like scaling and encoding based on the data types and code blocks, do only the necessary steps as per the given code blocks."""
    try:
        # Encoding "Position" column
        l = LabelEncoder()
        df["Position"] = l.fit_transform(df["Position"])
        logging.info("Encoded 'Position' column successfully.")

        # No scaling is required as per the given code blocks

        # No feature engineering required as per the given code blocks

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
    except FileNotFoundError as e:
        logging.error(f"Error loading model: Model file not found at {model_path}")
        exit()
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

def model_evaluation(model, X_test, y_test, y):
    """Evaluate model performance using MSE and R2."""
    try:
        # Predicting the target variable
        y_pred = model.predict(X_test)
        
        if y.dtype == "object":  # Classification case (if target is categorical) (Use all classification metrics)
            # Use all 5 classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1-Score: {f1}")
            logging.info(f"Confusion Matrix:\n{cm}")
        else:  # Regression case
            # Calculate regression metrics, use all regression metrics
            # Use all 5 regression metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            # Log and print the regression evaluation results
            logging.info(f"MAE: {mae}")
            logging.info(f"MSE: {mse}")
            logging.info(f"RMSE: {rmse}")
            logging.info(f"R²: {r2}")
            logging.info(f"MAPE: {mape}")
            
            return mse, r2
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        exit()

def main():
    # Load dataset (Replace 'dataset.csv' with actual dataset file if required)
    df = load_dataset("./Position_Salaries.csv") # Always load dataset from the same directory as the script ./

    # Preprocess data
    df = preprocess_data(df)

    # Prepare features and target variable based on the code blocks
    X = df[["Position", "Level"]]
    y = df["Salary"]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Load trained model (Replace 'model.pkl' with actual trained model file based on given code blocks)
    model = load_model("./finalized_model.pickle")  # Always load model from the same directory as the script ./

    # Run model inference on a sample row
    sample_row = X_test.sample(1)  # Select one row for prediction based on given code blocks
    prediction = model_inference(model, sample_row.values)

    # Evaluate the model (All evaluation metrics based on the target variable type)
    mse, r2 = model_evaluation(model, X_test, y_test, y)

    # Print results
    logging.info(f"Sample Prediction: {prediction}")
    logging.info(f"Model Evaluation - MSE: {mse}, R2: {r2}")

if __name__ == "__main__":
    main()
