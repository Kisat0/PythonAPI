from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
except (TypeError, KeyError) as e:
    print("Pipeline test failed:", e)

@app.get("/")
def read_root():
    return {"message": "API de prédiction des fermetures de restaurants"}
# Route de prédiction
@app.post("/predict")
def predict(request: PredictRequest):
    # Log the received features
    logger.info("Received features: %s", request.features)

    # Validate the input features
    if len(request.features) != 5:
        raise HTTPException(status_code=422, detail="Invalid number of features. Expected 5.")
    
    try:
        # Convert the input features to a DataFrame
        input_data = pd.DataFrame([request.features], columns=['LONGITUDE', 'LATITUDE', 'SCORE', 'CRITICAL FLAG', 'CUISINE DESCRIPTION'])
        logger.info("Input data reshaped for prediction: %s", input_data)
        
        # Make predictions
        prediction = model.predict(input_data)
        logger.info("Prediction result: %s", prediction)
        return {"prediction": prediction.tolist(), "probability": model.predict_proba(input_data).tolist()}
    except ValueError as ve:
        logger.error("ValueError during prediction: %s", str(ve))
        raise HTTPException(status_code=400, detail=f"ValueError: {str(ve)}") from ve
    except (TypeError, KeyError) as e:
        logger.error("An error occurred during prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}") from e
    

# Route de healthcheck du modèle
@app.get("/healthcheck")
def healthcheck():
    # Vérifier que le modèle est bien chargé
    try:
        healthcheck_sample_data = pd.DataFrame([[40.7128, -74.0060, 10, 1, 'Italian']],
                                               columns=['LONGITUDE', 'LATITUDE', 'SCORE', 'CRITICAL FLAG', 'CUISINE DESCRIPTION'])
        model.predict(healthcheck_sample_data)
        model_status = "Model is loaded and working"
    except (TypeError, KeyError, ValueError) as e:
        model_status = f"Model loading or prediction failed: {str(e)}"
    
    return {
        "status": "Healthcheck passed",
        "model_status": model_status,
    }
    

# Route de l'explication de la prédiction
@app.post("/explain")
def explain(request: PredictRequest):
    # Log the received features
    logger.info("Received features for explanation: %s", request.features)

    # Validate the input features
    if len(request.features) != 5:
        raise HTTPException(status_code=422, detail="Invalid number of features. Expected 5.")
    
    try:
        # Convert the input features to a DataFrame
        input_data = pd.DataFrame([request.features], columns=['LONGITUDE', 'LATITUDE', 'SCORE', 'CRITICAL FLAG', 'CUISINE DESCRIPTION'])
        logger.info("Input data reshaped for explanation: %s", input_data)
        
        # Explain the prediction
        explanation = model.explain(input_data)
        logger.info("Explanation result: %s", explanation)
        return {"explanation": explanation}
    except ValueError as ve:
        logger.error("ValueError during explanation: %s", str(ve))
        raise HTTPException(status_code=400, detail=f"ValueError: {str(ve)}") from ve
    except (TypeError, KeyError) as e:
        logger.error("An error occurred during explanation: %s", str(e))
        raise HTTPException(status_code=500, detail=f"An error occurred during explanation: {str(e)}") from e