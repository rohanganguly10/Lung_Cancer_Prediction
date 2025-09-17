# data_api.py
from fastapi import APIRouter
import pandas as pd

router = APIRouter()

# Load dataset
df = pd.read_csv("survey lung cancer.csv")
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

@router.get("/summary")
def dataset_summary():
    return {
        "records": len(df),
        "features": len(df.columns)-1,
        "positive_rate": round((df["LUNG_CANCER"].value_counts().get("YES",0)/len(df)*100),1)
    }

@router.get("/feature/{col}")
def feature_distribution(col: str):
    col = col.upper()
    if col not in df.columns or col == "LUNG_CANCER":
        return {"error": "Invalid column"}
    
    grouped = df.groupby([col, "LUNG_CANCER"]).size().reset_index(name="count")
    return grouped.to_dict(orient="records")
