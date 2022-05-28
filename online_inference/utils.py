import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from pydantic import BaseModel
from typing import List
import os

BASE_DIR = Path(__file__).resolve(strict=True).parent
DATA_PATH = '/online_inference/test_data/test_data.json'
MODEL_PATH = 'C:\\Users\\anke\\PycharmProjects\\ml_project\\online_inference\\models\\DecisionTreeClassifier.pickle'


class Predictions(BaseModel):
    index: int
    prediction: int


def predict(model_path, data_path) -> List[Predictions]:

    # with open(Path(BASE_DIR).joinpath(f'{data_path}.json'), 'r') as f:
    with open(data_path, 'r') as f:
        test_data = pd.read_json(f.read())

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    index = list(test_data.index)
    preds = model.predict(test_data)
    return [Predictions(index=ind, prediction=pred) for ind, pred in zip(index, preds)]

