import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Загрузка модели
with open('model.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

app = FastAPI()

# Определение модели для входных данных
class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)  # Преобразование в нужный формат
        prediction = loaded_pipeline.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}