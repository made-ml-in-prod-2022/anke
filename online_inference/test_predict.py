import pytest
import requests
from fastapi.testclient import TestClient
import pandas as pd
from fast import app
from test_data import fake_data

client = TestClient(app)


def test_unicorn():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == 'Do unicorns suffer from heart disease?'


@pytest.fixture
def get_fake_data():
    data = fake_data.make_fake_test_dataset(10)
    test_data = []
    for i in data.index:
        test_data.append([*data.iloc[i]])
    return test_data


def test_model_can_predict(get_fake_data):
    response = requests.get(
        'http://127.0.0.1:8000/predict/',
        json={'test_data': get_fake_data},
    )
    predictions = pd.DataFrame(response.json())
    assert len(predictions) == 10
    assert list(predictions.columns) == ['index', 'prediction']
    assert list(predictions['index']) == list(range(10))
    assert all([predictions.prediction.iloc[i] in [0, 1] for i in range(10)])
    assert response.status_code == 200



