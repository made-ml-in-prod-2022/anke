from data import Dataset, make_fake_dataset
from params import PreproParams, SplitParams
from pathlib import Path
import numpy as np
import os
import pytest

ROOT_PATH = Path(__file__).parent.parent
DATASET_PATH = os.path.join(ROOT_PATH, 'data/raw/heart_cleveland_upload.csv')
TARGET = 'condition'
COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                   'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']
ONE_HOT = ['cp', 'slope', 'thal', 'restecg']
CORR = ['age', 'exang', 'chol', 'fbs', 'thalach']
NORM = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

split_params_1 = SplitParams(split=True, test_size=0.25, random_state=1234, shuffle=False, target_column='condition')
split_params_2 = SplitParams(split=True, test_size=0.25, random_state=1234, shuffle=True, target_column='condition')
split_params_3 = SplitParams(split=True, test_size=0.5, random_state=0, shuffle=False, target_column='age')
split_params_default = SplitParams()


prepro_params_one_hot_half = PreproParams(feature_names=COLUMNS, correlated_features=CORR, one_hot_features=ONE_HOT[:2],
                                     norm_features=NORM, target_column='condition', transformer_preprocess=False,
                                     delete_correlated=False, norm=False, pca=False)
prepro_params_one_hot_zero = PreproParams(feature_names=COLUMNS, correlated_features=CORR, one_hot_features=[],
                                     norm_features=NORM, target_column='condition', transformer_preprocess=False,
                                     delete_correlated=False, norm=False, pca=False)

prepro_params_del_corr_half = PreproParams(feature_names=COLUMNS, correlated_features=CORR[:3], one_hot_features=[],
                                     norm_features=NORM, target_column='condition', transformer_preprocess=False,
                                     delete_correlated=True, norm=False, pca=False)

prepro_params_norm_half = PreproParams(feature_names=COLUMNS, correlated_features=CORR, one_hot_features=[],
                                     norm_features=NORM[:3], target_column='condition', transformer_preprocess=False,
                                     delete_correlated=False, norm=True, pca=False)

prepro_params_half = PreproParams(feature_names=COLUMNS, correlated_features=CORR[:3], one_hot_features=ONE_HOT[:2],
                                     norm_features=NORM[:3], target_column='condition', transformer_preprocess=False,
                                     delete_correlated=True, norm=True, pca=False)

prepro_params_pca_10 = PreproParams(feature_names=COLUMNS, correlated_features=CORR, one_hot_features=ONE_HOT,
                                     norm_features=NORM, target_column='condition', transformer_preprocess=False,
                                     delete_correlated=True, norm=True, pca=True, pca_n_comp=10)

prepro_params_pca_11 = PreproParams(feature_names=COLUMNS, correlated_features=CORR, one_hot_features=ONE_HOT,
                                     norm_features=NORM, target_column='condition', transformer_preprocess=False,
                                     delete_correlated=True, norm=True, pca=True, pca_n_comp=11)

prepro_params_all = PreproParams(feature_names=COLUMNS, correlated_features=CORR, one_hot_features=ONE_HOT,
                                     norm_features=NORM, target_column='condition', transformer_preprocess=False,
                                     delete_correlated=True, norm=True, pca=False)

prepro_params_transformer = PreproParams(feature_names=COLUMNS, correlated_features=CORR, one_hot_features=ONE_HOT,
                                     norm_features=NORM, target_column='condition', transformer_preprocess=True,
                                     delete_correlated=True, norm=True, pca=False)


@pytest.fixture()
def fake_dataset_300_path(tmpdir):
    full_path = make_fake_dataset(300, str(tmpdir))
    return full_path


def test_can_read_fake_csv(fake_dataset_300_path):
    dataset = Dataset(fake_dataset_300_path)
    assert dataset.data.shape == (300, 14)
    assert TARGET in dataset.data.columns
    assert sorted(COLUMNS) == sorted(list(dataset.data.columns))


def test_can_read_true_csv():
    dataset = Dataset(DATASET_PATH)
    assert dataset.data.shape == (297, 14)
    assert TARGET in dataset.data.columns
    assert sorted(COLUMNS) == sorted(list(dataset.data.columns))
    assert 69 == dataset.data.iloc[0]['age']
    assert 174 == dataset.data.iloc[3]['thalach']


@pytest.fixture()
def dataset_raw():
    return Dataset(DATASET_PATH)


@pytest.fixture()
def dataset_fake_300(fake_dataset_300_path):
    return Dataset(fake_dataset_300_path)


@pytest.mark.parametrize(
    "dataset", ["dataset_raw", "dataset_fake_300"]
)
def test_can_split(dataset, request):
    dataset = request.getfixturevalue(dataset)
    X_train_default, X_test_default, y_train_default, y_test_default = dataset.split(split_params_default)
    X_train_1, X_test_1, y_train_1, y_test_1 = dataset.split(split_params_1)
    X_train_2, X_test_2, y_train_2, y_test_2 = dataset.split(split_params_2)
    X_train_3, X_test_3, y_train_3, y_test_3 = dataset.split(split_params_3)
    assert (int(0.75 * len(dataset.data)), 13) == X_train_default.shape
    assert (int(0.75 * len(dataset.data)), ) == y_train_default.shape
    assert X_train_default.shape == X_train_1.shape
    assert list(X_train_default.index) == list(X_train_1.index)
    assert TARGET not in X_train_1.columns
    assert 'age' not in X_train_3.columns
    assert X_train_default.shape != X_train_3.shape
    assert (int(0.5 * len(dataset.data)), 13) == X_train_3.shape
    assert list(X_train_default.index) != list(X_train_2.index)


@pytest.mark.parametrize(
    "dataset", ["dataset_raw", "dataset_fake_300"]
)
def test_can_reproduce_splits(dataset, request):
    dataset = request.getfixturevalue(dataset)
    X_train_1, X_test_1, y_train_1, y_test_1 = dataset.split(split_params_1)
    X_train_2, X_test_2, y_train_2, y_test_2 = dataset.split(split_params_1)
    assert X_train_1.shape == X_train_2.shape
    assert list(X_train_1.index) == list(X_train_2.index)
    assert list(X_train_1.index) == list(y_train_1.index)
    assert list(X_train_1.columns) == list(X_train_2.columns)
    assert all(list(X_train_1[feature]) == list(X_train_2[feature]) for feature in X_train_1.columns)
    assert list(y_train_1) == list(y_train_2)
    assert list(y_test_1) == list(y_test_2)


@pytest.mark.parametrize(
    "dataset", ["dataset_raw", "dataset_fake_300"]
)
def test_can_get_features_and_target(dataset, request):
    dataset = request.getfixturevalue(dataset)
    feats, target = dataset.get_data_and_target(split_params_3)
    assert 'age' not in feats.columns
    assert target.name == 'age'
    assert len(feats.columns) == 13


@pytest.mark.parametrize(
    "dataset", ["dataset_raw", "dataset_fake_300"]
)
def test_can_prepro_one_hot(dataset, request):
    dataset = request.getfixturevalue(dataset)
    prev_shape = dataset.data.shape
    dataset.preprocess(prepro_params_one_hot_zero)
    assert dataset.data.shape == prev_shape
    dataset.preprocess(prepro_params_one_hot_half)
    assert dataset.data.shape != prev_shape
    assert any(feature not in dataset.data.columns for feature in ONE_HOT[:2])
    assert all(feature in dataset.data.columns for feature in ['cp__0', 'cp__1', 'slope__0', 'slope__1'])
    assert all(feature in dataset.data.columns for feature in ['thal', 'restecg'])
    assert any(feature not in dataset.data.columns for feature in ['thal__0', 'restecg__0'])


@pytest.mark.parametrize(
    "dataset", ["dataset_raw", "dataset_fake_300"]
)
def test_can_delete_correlated(dataset, request):
    dataset = request.getfixturevalue(dataset)
    prev_cols_len = len(dataset.data.columns)
    dataset.preprocess(prepro_params_del_corr_half)
    assert prev_cols_len == len(dataset.data.columns) + 3
    assert any(feature not in dataset.data.columns for feature in CORR[:3])
    assert all(feature in dataset.data.columns for feature in CORR[3:])


@pytest.mark.parametrize(
    "dataset", ["dataset_raw", "dataset_fake_300"]
)
def test_can_normalize(dataset, request):
    dataset = request.getfixturevalue(dataset)
    prev_shape = dataset.data.shape
    dataset.preprocess(prepro_params_norm_half)
    assert prev_shape == dataset.data.shape
    assert all(np.allclose(dataset.data[feature].mean(), 0) for feature in NORM[:3])
    assert all(np.allclose(dataset.data[feature].var(), 1) for feature in NORM[:3])


@pytest.mark.parametrize(
    "dataset", ["dataset_raw", "dataset_fake_300"]
)
def test_can_one_hot_delete_and_norm(dataset, request):
    dataset = request.getfixturevalue(dataset)
    dataset.preprocess(prepro_params_half)
    assert any(feature not in dataset.data.columns for feature in ONE_HOT[:2])
    assert all(feature in dataset.data.columns for feature in ['cp__0', 'cp__1', 'slope__0', 'slope__1'])
    assert all(feature in dataset.data.columns for feature in ['thal', 'restecg'])
    assert any(feature not in dataset.data.columns for feature in ['thal__0', 'restecg__0'])
    assert any(feature not in dataset.data.columns for feature in CORR[:3])
    assert all(feature in dataset.data.columns for feature in CORR[3:])
    assert np.allclose(dataset.data[NORM[1]].mean(), 0)
    assert np.allclose(dataset.data[NORM[1]].var(), 1)


def test_can_apply_pca():
    dataset = Dataset(DATASET_PATH)
    dataset.preprocess(prepro_params_pca_10)
    dataset.apply_pca(prepro_params_pca_10)
    data_10 = dataset.data.copy()

    dataset = Dataset(DATASET_PATH)
    dataset.preprocess(prepro_params_pca_11)
    dataset.apply_pca(prepro_params_pca_11)
    data_11 = dataset.data.copy()
    assert (297, 11) == data_10.shape
    assert (297, 12) == data_11.shape
    assert all(np.allclose(data_11[i], data_10[i]) for i in range(10))


@pytest.fixture()
def dataset_true_preprocessed():
    dataset = Dataset(DATASET_PATH)
    dataset.preprocess(prepro_params_all)
    return dataset


@pytest.fixture()
def dataset_fake_preprocessed(fake_dataset_300_path):
    dataset = Dataset(fake_dataset_300_path)
    dataset.preprocess(prepro_params_all)
    return dataset


@pytest.mark.parametrize(
    "dataset, dataset_preprocessed", [("dataset_raw", "dataset_true_preprocessed"),
                                      ("dataset_fake_300", "dataset_fake_preprocessed")]
)
def test_can_preprocess_with_transformer(dataset, dataset_preprocessed, request):
    dataset = request.getfixturevalue(dataset)
    dataset_preprocessed = request.getfixturevalue(dataset_preprocessed)
    dataset.preprocess_with_transformer(prepro_params_transformer)
    data_transf = dataset.data
    data_prepro = dataset_preprocessed.data
    assert data_transf.shape == data_prepro.shape
    # немного не совпадают из-за разных нормализаций (rtol)
    assert all(np.allclose(data_prepro[column], data_transf[column], rtol=1e-2) for column in data_prepro.columns)




