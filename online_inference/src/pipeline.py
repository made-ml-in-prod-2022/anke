import pandas as pd
import pickle
from pydantic import BaseModel
from typing import List, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from .inference_params import read_inference_params, InferenceParams

DISEASE_ESTIMATORS = [RandomForestClassifier, DecisionTreeClassifier, LogisticRegression]


class Predictions(BaseModel):
    index: int
    prediction: int


class ModelHealth(BaseModel):
    is_okay: bool


def preprocess(params: InferenceParams) -> pd.DataFrame:
    with open(params.data_path, 'r') as f:
        data = pd.read_json(f)
    # delete correlated
    columns = sorted(list(set(params.feature_names) - set(params.correlated_features)))
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


def load_model(cfg_path: str):
    inference_params = read_inference_params(cfg_path)
    with open(inference_params.model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def check_model(model):
    check = any([isinstance(model, estimator) for estimator in DISEASE_ESTIMATORS])
    if check:
        check = check_is_fitted(model) is None
    return ModelHealth(is_okay=check)


def predict(model, test_data: pd.DataFrame) -> List[Predictions]:
    index = list(test_data.index)
    preds = model.predict(test_data)
    return [Predictions(index=ind, prediction=pred) for ind, pred in zip(index, preds)]


def inference_pipeline(cfg_path: str) -> List[Predictions]:
    inference_params = read_inference_params(cfg_path)
    test_data_preprocessed = preprocess(inference_params)
    model = load_model(cfg_path)
    predictions = predict(model, test_data_preprocessed)
    return predictions


