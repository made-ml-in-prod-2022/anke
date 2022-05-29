from fastapi import FastAPI
from typing import List

import uvicorn
from src import inference_pipeline, Predictions, ModelHealth, load_model, check_model, TestData

CONFIG_PATH = 'config.yaml'

app = FastAPI()


@app.get('/')
def hello():
    return 'Do unicorns suffer from heart disease?'


@app.get('/predict/', response_model=List[Predictions], status_code=200)
async def predict_disease(request: TestData):
    preds = inference_pipeline(request.test_data, CONFIG_PATH)
    return preds


@app.get('/health/', response_model=ModelHealth, status_code=200)
async def is_ready():
    model = load_model(CONFIG_PATH)
    response = check_model(model)
    return response

if __name__ == "__main__":
    uvicorn.run("fast:app", host="127.0.0.1", port=8000)
