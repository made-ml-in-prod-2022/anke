from fastapi import FastAPI
from fastapi.testclient import TestClient
import os
from pathlib import Path
from online_inference import app

ROOT_PATH = Path(__file__).parent.parent
ORIGINAL_DATA_PATH = os.path.join(ROOT_PATH, 'test_data/heart_cleveland_upload.csv')

client = TestClient(app)


def test_unicorn():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json == {'Do unicorns suffer from heart disease?'}

