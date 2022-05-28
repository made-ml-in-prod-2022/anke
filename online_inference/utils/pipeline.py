import pandas as pd
import pickle
from pathlib import Path
from pydantic import BaseModel
from typing import List
from .inference_params import read_inference_params, InferenceParams


BASE_DIR = Path(__file__).resolve(strict=True).parent


class Predictions(BaseModel):
    index: int
    prediction: int


def preprocess(params: InferenceParams) -> pd.DataFrame:
    with open(params.data_path, 'r') as f:
        data = pd.read_json(f)
    # delete correlated
    columns = sorted(list(set(params.feature_names) - set(params.correlated_features)))
    print(data.head())
    data = data[columns]
    # one-hot
    one_hot = sorted(list(set(params.one_hot_features) & set(columns)))
    for col in one_hot:
        dummies = pd.get_dummies(data[col], prefix=f'{col}_')
        data = data.join(dummies)
    data.drop(columns=one_hot, inplace=True)
    # normalize
    to_norm = sorted(list(set(params.norm_features) & set(columns)))
    data[to_norm] = ((data[to_norm] - data[to_norm].mean(axis=0)) /
                     (data[to_norm].var(axis=0)) ** (1 / 2))

    return data


def predict(params: InferenceParams, test_data: pd.DataFrame) -> List[Predictions]:
    # with open(Path(BASE_DIR).joinpath(f'{data_path}.json'), 'r') as f:
    with open(params.model_path, 'rb') as f:
        model = pickle.load(f)
    index = list(test_data.index)
    preds = model.predict(test_data)
    return [Predictions(index=ind, prediction=pred) for ind, pred in zip(index, preds)]


def inference_pipeline(cfg_path: str) -> List[Predictions]:
    inference_params = read_inference_params(cfg_path)
    test_data_preprocessed = preprocess(inference_params)
    predictions = predict(inference_params, test_data_preprocessed)
    return predictions

from sklearn.model_selection import train_test_split

with open('C:\\Users\\anke\\PycharmProjects\\ml_project\\online_inference\\test_data\\heart_cleveland_upload.csv', 'r') as f:
    data = pd.read_csv(f)
data.drop(columns='Unnamed: 0', inplace=True)
y = data.condition
x = data.drop(columns='condition')
x_train, x_test, y_train, y_test = train_test_split(x, y)
with open('x_train.csv') as f:
    x_train.to_json(f)


