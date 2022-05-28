from fastapi import FastAPI
from typing import Dict, List
import uvicorn
import os
import requests
import pandas as pd
from fastapi import Request
from utils import inference_pipeline, Predictions

CONFIG_PATH = 'C:\\Users\\anke\\PycharmProjects\\ml_project\\online_inference\\config.yaml'

app = FastAPI()


@app.get('/predict/', response_model=List[Predictions], status_code=200)
async def predict_disease():
    preds = inference_pipeline(CONFIG_PATH)
    return preds

if __name__ == "__main__":
    uvicorn.run("fast:app", host="127.0.0.1", port=8000)
