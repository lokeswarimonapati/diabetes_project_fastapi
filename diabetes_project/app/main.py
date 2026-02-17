from fastapi import FastAPI
import numpy as np
import joblib
import os

app = FastAPI()

MODEL_PATH = "model/diabetes_pipeline.pkl"

# Load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.get("/")
def home():
    return {"message": "Diabetes API Running 🚀"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded"}
    
    input_data = np.array([[ 
        data["Pregnancies"],
        data["Glucose"],
        data["BloodPressure"],
        data["SkinThickness"],
        data["Insulin"],
        data["BMI"],
        data["DiabetesPedigreeFunction"],
        data["Age"]
    ]])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": int(prediction),
        "result": "Diabetic" if prediction == 1 else "Not Diabetic"
    }
