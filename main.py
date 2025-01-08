from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialiser l'application
app = FastAPI()

# Charger le modèle
MODEL_PATH = "models/restaurant_closures_rf.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier modèle {MODEL_PATH} est introuvable.")

model = joblib.load(MODEL_PATH)

# Définir la structure des données entrantes
class PredictRequest(BaseModel):
    features: list[float]  # Liste des caractéristiques pour la prédiction


@app.get("/")
def read_root():
    return {"message": "API de prédiction des fermetures de restaurants"}

# Route de prédiction
@app.post("/predict")
def predict(request: PredictRequest):
    # Vérifier que les données sont au bon format
    if not isinstance(request.features, list) or len(request.features) == 0:
        raise HTTPException(status_code=400, detail="Les 'features' doivent être une liste non vide.")

    # Convertir les données en tableau numpy
    input_data = np.array(request.features).reshape(1, -1)

    try:
        # Faire la prédiction
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")
