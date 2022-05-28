from fastapi import FastAPI
from typing import Dict, List
import uvicorn
import os
import requests
import pandas as pd
from fastapi import Request
from utils import predict, MODEL_PATH, DATA_PATH, Predictions

app = FastAPI()


@app.get('/predict/', response_model=List[Predictions], status_code=200)
async def predict_desease():
    # input_data = data.data

    # input_df = pd.DataFrame([input_data])
    # print(input_df.head())

    preds = predict(MODEL_PATH, DATA_PATH)
    return preds

if __name__ == "__main__":
    uvicorn.run("fast:app", host="127.0.0.1", port=8000)
