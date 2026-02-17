import joblib
import os

MODEL_PATH = "model/diabetes_pipeline.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found!")
    return joblib.load(MODEL_PATH)
