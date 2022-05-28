import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from models import Model
from params import TrainParams
from data_tests import dataset_true_preprocessed, dataset_fake_preprocessed, fake_dataset_300_path
from params import SplitParams
import os
import pickle


split_params = SplitParams(split=True, test_size=0.25, random_state=1234, shuffle=False, target_column='condition')

rf_params = TrainParams(model_type='RandomForestClassifier', random_state=1, max_depth=5, min_samples_leaf=3,
                        max_features='sqrt')

tree_params = TrainParams(model_type='DecisionTreeClassifier', random_state=2, max_depth=8, min_samples_leaf=2,
                          max_features='sqrt')

logit_params = TrainParams(model_type='LogisticRegression', random_state=3, penalty='l1', solver='liblinear',
                           reg_param=0.5)

wrong_params = TrainParams(model_type='SVM', random_state=3, penalty='l1', solver='liblinear',
                           reg_param=0.5)


@pytest.fixture()
def splitted_true_preprocessed_data(dataset_true_preprocessed):
    X_train, X_test, y_train, y_test = dataset_true_preprocessed.split(split_params)
    return X_train, X_test, y_train, y_test


@pytest.fixture()
def splitted_fake_preprocessed_data(dataset_fake_preprocessed):
    X_train, X_test, y_train, y_test = dataset_fake_preprocessed.split(split_params)
    return X_train, X_test, y_train, y_test


def test_can_init_rf():
    model = Model(rf_params)
    estimator_params = model.estimator.get_params()
    assert model.name == 'RandomForestClassifier'
    assert model.model_type == 'RandomForestClassifier'

    assert isinstance(model.estimator, RandomForestClassifier)
    assert estimator_params['max_depth'] == rf_params.max_depth
    assert estimator_params['random_state'] == rf_params.random_state


def test_can_init_tree():
    model = Model(tree_params)
    estimator_params = model.estimator.get_params()
    assert model.name == 'DecisionTreeClassifier'
    assert model.model_type == 'DecisionTreeClassifier'
    assert isinstance(model.estimator, DecisionTreeClassifier)
    assert estimator_params['max_depth'] == tree_params.max_depth
    assert estimator_params['random_state'] == tree_params.random_state


def test_can_init_logit():
    model = Model(logit_params)
    estimator_params = model.estimator.get_params()
    assert model.name == 'LogisticRegression'
    assert model.model_type == 'LogisticRegression'
    assert isinstance(model.estimator, LogisticRegression)
    assert estimator_params['random_state'] == logit_params.random_state
    assert estimator_params['C'] == logit_params.reg_param


def test_cant_init_smth_else():
    try:
        model = Model(wrong_params)
    except NotImplementedError:
        pass


@pytest.mark.parametrize(
    "splitted_preprocessed_data", ["splitted_true_preprocessed_data", "splitted_fake_preprocessed_data"]
)
def test_can_train(splitted_preprocessed_data, request):
    splitted_preprocessed_data = request.getfixturevalue(splitted_preprocessed_data)
    X_train, y_train = splitted_preprocessed_data[0], splitted_preprocessed_data[2]
    rf_model = Model(rf_params)
    rf_model.train(X_train, y_train)
    tree_model = Model(tree_params)
    tree_model.train(X_train, y_train)
    logit_model = Model(logit_params)
    logit_model.train(X_train, y_train)
    assert check_is_fitted(rf_model.estimator) is None
    assert check_is_fitted(tree_model.estimator) is None
    assert check_is_fitted(logit_model.estimator) is None


@pytest.fixture()
def fitted_rf(splitted_true_preprocessed_data):
    X_train, y_train = splitted_true_preprocessed_data[0], splitted_true_preprocessed_data[2]
    rf_model = Model(rf_params)
    rf_model.train(X_train, y_train)
    return rf_model


@pytest.fixture()
def fitted_fake_rf(splitted_fake_preprocessed_data):
    X_train, y_train = splitted_fake_preprocessed_data[0], splitted_fake_preprocessed_data[2]
    rf_model = Model(rf_params)
    rf_model.train(X_train, y_train)
    return rf_model


@pytest.fixture()
def fitted_tree(splitted_true_preprocessed_data):
    X_train, y_train = splitted_true_preprocessed_data[0], splitted_true_preprocessed_data[2]
    tree_model = Model(tree_params)
    tree_model.train(X_train, y_train)
    return tree_model


@pytest.fixture()
def fitted_logit(splitted_true_preprocessed_data):
    X_train, y_train = splitted_true_preprocessed_data[0], splitted_true_preprocessed_data[2]
    logit_model = Model(logit_params)
    logit_model.train(X_train, y_train)
    return logit_model


@pytest.fixture()
def fitted_svm(splitted_true_preprocessed_data):
    X_train, y_train = splitted_true_preprocessed_data[0], splitted_true_preprocessed_data[2]
    svm_estimator = SVC()
    svm_estimator.fit(X_train, y_train)
    return svm_estimator


def test_can_serialize(fitted_rf, tmpdir, splitted_true_preprocessed_data):
    X_train = splitted_true_preprocessed_data[0]
    ckpt_tmp = str(tmpdir)
    full_path = fitted_rf.serialize_estimator(ckpt_tmp)
    with open(full_path, 'rb') as file:
        loaded_rf = pickle.load(file)
    assert full_path == os.path.join(ckpt_tmp, fitted_rf.name + '.pickle')
    rf_train_preds = fitted_rf.estimator.predict(X_train)
    loaded_rf_train_preds = loaded_rf.predict(X_train)
    assert os.path.exists(full_path)
    assert isinstance(loaded_rf, RandomForestClassifier)
    assert loaded_rf.get_params() == fitted_rf.estimator.get_params()
    assert check_is_fitted(loaded_rf) is None
    assert list(loaded_rf_train_preds) == list(rf_train_preds)


@pytest.fixture()
def serialized_rf_tmp_path(fitted_rf, tmpdir):
    ckpt_tmp = str(tmpdir)
    full_path = fitted_rf.serialize_estimator(ckpt_tmp)
    return full_path


@pytest.fixture()
def serialized_tree_tmp_path(fitted_tree, tmpdir):
    ckpt_tmp = str(tmpdir)
    full_path = fitted_tree.serialize_estimator(ckpt_tmp)
    return full_path


@pytest.fixture()
def serialized_logit_tmp_path(fitted_logit, tmpdir):
    ckpt_tmp = str(tmpdir)
    full_path = fitted_logit.serialize_estimator(ckpt_tmp)
    return full_path


@pytest.fixture()
def serialized_svm_tmp_path(fitted_svm, tmpdir):
    ckpt_tmp = str(tmpdir)
    full_path = os.path.join(ckpt_tmp, 'svm' + '.pickle')
    with open(full_path, "wb") as f:
        pickle.dump(fitted_svm, f)
    return full_path


def test_can_init_rf_from_serialized(serialized_rf_tmp_path, fitted_rf):
    rf_serialized_params = TrainParams(eval_from_path=True, ckpt_path=serialized_rf_tmp_path)
    rf_from_ckpt = Model(rf_serialized_params)
    assert rf_from_ckpt.name == 'RandomForestClassifier'
    assert rf_from_ckpt.model_type == 'RandomForestClassifier'
    assert isinstance(rf_from_ckpt.estimator, RandomForestClassifier)
    assert check_is_fitted(rf_from_ckpt.estimator) is None
    assert rf_from_ckpt.estimator.get_params() == fitted_rf.estimator.get_params()


def test_can_init_tree_from_serialized(serialized_tree_tmp_path, fitted_tree):
    tree_serialized_params = TrainParams(eval_from_path=True, ckpt_path=serialized_tree_tmp_path)
    tree_from_ckpt = Model(tree_serialized_params)
    assert tree_from_ckpt.name == 'DecisionTreeClassifier'
    assert tree_from_ckpt.model_type == 'DecisionTreeClassifier'
    assert isinstance(tree_from_ckpt.estimator, DecisionTreeClassifier)
    assert check_is_fitted(tree_from_ckpt.estimator) is None
    assert tree_from_ckpt.estimator.get_params() == fitted_tree.estimator.get_params()


def test_can_init_logit_from_serialized(serialized_logit_tmp_path, fitted_logit):
    logit_serialized_params = TrainParams(eval_from_path=True, ckpt_path=serialized_logit_tmp_path)
    logit_from_ckpt = Model(logit_serialized_params)
    assert logit_from_ckpt.name == 'LogisticRegression'
    assert logit_from_ckpt.model_type == 'LogisticRegression'
    assert isinstance(logit_from_ckpt.estimator, LogisticRegression)
    assert check_is_fitted(logit_from_ckpt.estimator) is None
    assert logit_from_ckpt.estimator.get_params() == fitted_logit.estimator.get_params()


def test_cant_init_wrong_model(serialized_svm_tmp_path, fitted_svm):
    svm_serialized_params = TrainParams(eval_from_path=True, ckpt_path=serialized_svm_tmp_path)
    try:
        svm_from_ckpt = Model(svm_serialized_params)
    except NotImplementedError:
        pass


@pytest.mark.parametrize(
    "rf, splitted_preprocessed_data", [("fitted_rf", "splitted_true_preprocessed_data"),
                                       ("fitted_fake_rf", "splitted_fake_preprocessed_data")]
)
def test_can_predict(rf, splitted_preprocessed_data, request):
    rf = request.getfixturevalue(rf)
    splitted_preprocessed_data = request.getfixturevalue(splitted_preprocessed_data)
    new_rf = RandomForestClassifier(random_state=rf_params.random_state, max_depth=rf_params.max_depth,
                                    min_samples_leaf=rf_params.min_samples_leaf,
                                    max_features=rf_params.max_features)
    X_train, X_test, y_train = splitted_preprocessed_data[0], \
                               splitted_preprocessed_data[1], splitted_preprocessed_data[2]
    new_rf.fit(X_train, y_train)
    new_preds = new_rf.predict(X_test)
    model_preds = rf.predict(X_test)
    assert list(model_preds) == list(new_preds)


def test_can_dump_preds(tmpdir, fitted_rf, splitted_true_preprocessed_data):
    X_test = splitted_true_preprocessed_data[1]
    preds = fitted_rf.predict(X_test)
    full_path = fitted_rf.dump_preds(preds, str(tmpdir))
    with open(full_path, 'rb') as file:
        loaded_preds = pickle.load(file)
    assert full_path == os.path.join(str(tmpdir), fitted_rf.name + '_preds.pickle')
    assert list(loaded_preds) == list(preds)


@pytest.mark.parametrize(
    "rf, splitted_preprocessed_data", [("fitted_rf", "splitted_true_preprocessed_data"),
                                       ("fitted_fake_rf", "splitted_fake_preprocessed_data")]
)
def test_can_eval(rf, splitted_preprocessed_data, request):
    rf = request.getfixturevalue(rf)
    splitted_preprocessed_data = request.getfixturevalue(splitted_preprocessed_data)
    X_test, y_test = splitted_preprocessed_data[1], splitted_preprocessed_data[3]
    preds = rf.predict(X_test)
    eval_res = rf.eval(preds, y_test)
    assert isinstance(eval_res, dict)
    assert all(metric in eval_res for metric in ['recall', 'precision', 'roc_auc_score'])


def test_can_dump_metrics(fitted_rf, tmpdir, splitted_true_preprocessed_data):
    X_test, y_test = splitted_true_preprocessed_data[1], splitted_true_preprocessed_data[3]
    preds = fitted_rf.predict(X_test)
    eval_res = fitted_rf.eval(preds, y_test)
    full_path = fitted_rf.dump_eval(eval_res, str(tmpdir))
    with open(full_path, 'rb') as file:
        loaded_evals = pickle.load(file)
    assert full_path == os.path.join(str(tmpdir), fitted_rf.name + '_eval_res.pickle')
    assert list(loaded_evals) == list(eval_res)







