

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

app = FastAPI()



# Reload the model
pipeline_filepath = "pipeline.joblib"
pipeline = joblib.load(pipeline_filepath)

# Save the model again
joblib.dump(pipeline, pipeline_filepath)


class PatientData(BaseModel):
    Plasma_glucose : float
    Blood_Work_Result_1: float
    Blood_Pressure : float
    Blood_Work_Result_2 : float
    Blood_Work_Result_3 : float
    Body_mass_index  : float
    Blood_Work_Result_4: float
    Age: float
    Insurance: int

@app.get("/")
def read_root():
    explanation = {
        'message': "Welcome to the Sepsis Prediction App",
        'description': "This API allows you to predict sepsis based on patient data.",
        'usage': "Submit a POST request to /predict with patient data to make predictions.",
        
    }
    return explanation



import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

@app.post("/predict")
def get_data_from_user(data: PatientData):
    try:
        logging.info("Received data: %s", data.dict())
        user_input = data.dict()
        input_df = pd.DataFrame([user_input])

        # Make predictions using the loaded pipeline
        prediction = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)

        probability_of_positive_class = probabilities[0][1]

        # Calculate the prediction
        sepsis_status = "Positive" if prediction[0] == 1 else "Negative"

        result = {
            'predicted_sepsis': sepsis_status,
            'probability': probability_of_positive_class,
        }
        return result
    except Exception as e:
        logging.error("Error: %s", e)
        return {"error": "An error occurred during prediction."}
