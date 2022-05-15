import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, precision_score,  recall_score

import pandas as pd
import numpy as np
from params import TrainParams
from typing import Dict


class Model:
    def __init__(self, params: TrainParams):
        # load serialized estimator
        # data preprocessing should match exactly
        if params.eval_from_path:
            with open(params.ckpt_path, "rb") as f:
                self.estimator = pickle.load(f)
            if isinstance(self.estimator, RandomForestClassifier):
                self.model_type = 'RandomForestClassifier'
            elif isinstance(self.estimator, DecisionTreeClassifier):
                self.model_type = 'DecisionTreeClassifier'
            elif isinstance(self.estimator, LogisticRegression):
                self.model_type = 'LogisticRegression'
            else:
                raise NotImplementedError

            ckpt_name = os.path.basename(params.ckpt_path)
            self.name = ckpt_name[:ckpt_name.find('.')]

        # create new estimator
        elif params.model_type == 'LogisticRegression':
            self.model_type = 'LogisticRegression'
            self.name = params.name
            self.estimator = LogisticRegression(penalty=params.penalty, solver=params.solver,
                                                C=params.reg_param, random_state=params.random_state)

        elif params.model_type == 'DecisionTreeClassifier' or params.model_type == 'RandomForestClassifier':
            self.name = params.name
            self.model_type = params.model_type

            if params.model_type == 'DecisionTreeClassifier':
                self.estimator = DecisionTreeClassifier(max_depth=params.max_depth,
                                                        min_samples_leaf=params.min_samples_leaf,
                                                        random_state=params.random_state)
            else:
                self.estimator = RandomForestClassifier(max_depth=params.max_depth,
                                                        min_samples_leaf=params.min_samples_leaf,
                                                        max_features=params.max_features,
                                                        random_state=params.random_state)
        else:
            raise NotImplementedError()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.estimator.fit(X_train, y_train)

    def serialize_estimator(self,  ckpt_path: str) -> str:
        full_path = os.path.join(ckpt_path, self.name + '.pickle')
        with open(full_path, "wb") as f:
            pickle.dump(self.estimator, f)
        return full_path

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X_test)

    def dump_preds(self, preds: np.ndarray, preds_path: str) -> str:
        full_path = os.path.join(preds_path, self.name + '_preds.pickle')
        with open(full_path, 'wb') as f:
            pickle.dump(preds, f)
        return full_path

    def eval(self, preds: np.ndarray, target: pd.Series) -> Dict:
        eval_results = {'recall': recall_score(target, preds), 'precision': precision_score(target, preds),
                        'roc_auc_score': roc_auc_score(target, preds)}
        return eval_results

    def dump_eval(self, eval_results: Dict, eval_path: str) -> str:
        full_path = os.path.join(eval_path, self.name + '_eval_res.pickle')
        with open(full_path, 'wb') as f:
            pickle.dump(eval_results, f)
        return full_path

