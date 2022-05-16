from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from params import PreproParams, SplitParams
from sklearn.decomposition import PCA
from dataclasses import dataclass


@dataclass()
class Dataset:
    path: str

    def __post_init__(self):
        self.data = pd.read_csv(self.path)

    def split(self, params: SplitParams) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Split data into train and test """
        y_data = self.data[params.target_column].copy()
        X_data = self.data.copy()
        X_data.drop(columns=[params.target_column], inplace=True)
        return train_test_split(X_data, y_data, random_state=params.random_state,
                                test_size=params.test_size, shuffle=params.shuffle)

    def get_data_and_target(self, params: SplitParams) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features from target. May be used for train or inference on full or new data"""
        y_data = self.data[params.target_column].copy()
        X_data = self.data.copy()
        X_data.drop(columns=[params.target_column], inplace=True)
        return X_data, y_data

    def preprocess(self, params: PreproParams) -> None:
        if params.delete_correlated:
            # sorting is for reproducibility
            columns = sorted(list(set(params.feature_names) - set(params.correlated_features)))
        else:
            columns = params.feature_names

        self.data = self.data[columns]

        one_hot = sorted(list(set(params.one_hot_features) & set(columns)))

        for col in one_hot:
            dummies = pd.get_dummies(self.data[col], prefix=f'{col}_')
            self.data = self.data.join(dummies)
        self.data.drop(columns=one_hot, inplace=True)

        if params.norm:
            to_norm = sorted(list(set(params.norm_features) & set(columns)))
            self.data[to_norm] = ((self.data[to_norm] - self.data[to_norm].mean(axis=0)) /
                                  (self.data[to_norm].var(axis=0)) ** (1 / 2))

        return

    def apply_pca(self, params: PreproParams) -> None:
        pca = PCA(n_components=params.pca_n_comp)
        target = self.data[params.target_column]
        self.data.drop(columns=[params.target_column], inplace=True)
        self.data = pd.DataFrame(pca.fit_transform(self.data))
        self.data[params.target_column] = target
        return

    def preprocess_with_transformer(self, params: PreproParams) -> None:
        """Applies both drop of correlated columns, normalization and one-hot of corresponding features"""
        # only full data
        target = self.data[params.target_column]
        self.data.drop(columns=[params.target_column], inplace=True)
        transformer = build_preprocessing_transformer(params)
        self.data = pd.DataFrame(transformer.fit_transform(self.data))

        self.data[params.target_column] = target
        self.data.columns = ['cp__0', 'cp__1', 'cp__2', 'cp__3', 'slope__0', 'slope__1', 'slope__2',
                             'thal__0', 'thal__1', 'thal__2', 'restecg__0', 'restecg__1', 'restecg__2',
                             'ca', 'oldpeak', 'trestbps', 'sex', 'condition']


def build_preprocessing_transformer(params: PreproParams) -> ColumnTransformer:
    cols_to_drop = sorted(list(set(params.one_hot_features) | set(params.norm_features)
                               | set(params.correlated_features)))  # 3й пайп
    cols_to_norm = sorted(list(set(params.norm_features) - set(params.correlated_features)))
    cols_to_pass = sorted(list(set(params.feature_names) - {params.target_column}))
    return ColumnTransformer(
        [
            ('ohe_pipeline', Pipeline([('ohe', OneHotEncoder())]), [*params.one_hot_features]),
            ('scaling_pipeline', Pipeline([('scaler', StandardScaler())]), [*cols_to_norm]),
            ('col_dropping_pipeline', Pipeline([
                ("selector", ColumnTransformer([
                    ("selector", "drop", [*cols_to_drop])],
                    remainder="passthrough"))]),
             [*cols_to_pass])
        ]
    )
