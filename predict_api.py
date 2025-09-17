from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np

router = APIRouter()

# Load ML models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Input model for request validation
class PatientData(BaseModel):
    YELLOW_FINGERS: str
    ANXIETY: str
    PEER_PRESSURE: str
    CHRONIC: str
    FATIGUE: str
    ALLERGY: str
    WHEEZING: str
    ALCOHOL: str
    COUGHING: str
    SWALLOWING: str
    CHEST_PAIN: str
    model: str = "Random Forest"

def encode(val: str) -> int:
    return 1 if val.lower() == "yes" else 0

@router.post("/predict")
def predict(data: PatientData):
    features = [
        encode(data.YELLOW_FINGERS),
        encode(data.ANXIETY),
        encode(data.PEER_PRESSURE),
        encode(data.CHRONIC),
        encode(data.FATIGUE),
        encode(data.ALLERGY),
        encode(data.WHEEZING),
        encode(data.ALCOHOL),
        encode(data.COUGHING),
        encode(data.SWALLOWING),
        encode(data.CHEST_PAIN)
    ]
    derived = features[0] * features[1]
    final_input = np.array([features + [derived]])

    model = rf_model if data.model == "Random Forest" else xgb_model
    pred = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][pred]

    return {
        "prediction": int(pred),
        "confidence": round(float(prob), 3),
        "model": data.model
    }
