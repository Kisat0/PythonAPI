from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from comet_ml.integration.sklearn import load_model
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List

# Initialiser le logger
logger = logging.getLogger(__name__)

origins = [
    "http://localhost",
    "http://localhost:5173",
]

# Initialiser l'application
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model from Comet
model = load_model("registry://n0ku/AI-la-Carte-Random_Forest-3.0")
print("Model loaded successfully:", model)

# Définir la structure des données entrantes
class PredictRequest(BaseModel):
    features: List

# Test the pipeline with sample data
sample_data = pd.DataFrame([[40.7128, -74.0060, 10, 1, 'Italian']],
            columns=['LONGITUDE', 'LATITUDE', 'SCORE', 'CRITICAL FLAG', 'CUISINE DESCRIPTION'])
try:
    model.predict(sample_data)
    print("Pipeline test successful")
except Exception as e:
    print("Pipeline test failed:", e)

@app.get("/")
def read_root():
    return {"message": "API de prédiction des fermetures de restaurants"}
# Route de prédiction
@app.post("/predict")
def predict(request: PredictRequest):
    # Log the received features
    logger.info(f"Received features: {request.features}")

    # Validate the input features
    if len(request.features) != 5:
        raise HTTPException(status_code=422, detail="Invalid number of features. Expected 5.")
    
    try:
        # Convert the input features to a DataFrame
        input_data = pd.DataFrame([request.features], columns=['LONGITUDE', 'LATITUDE', 'SCORE', 'CRITICAL FLAG', 'CUISINE DESCRIPTION'])
        logger.info(f"Input data reshaped for prediction: {input_data}")
        
        # Make predictions
        prediction = model.predict(input_data)
        logger.info(f"Prediction result: {prediction}")
        return {"prediction": prediction.tolist()}
    except ValueError as ve:
        logger.error(f"ValueError during prediction: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"ValueError: {str(ve)}")
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")