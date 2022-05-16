import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_PATH = Path(__file__).parent.parent
ORIGINAL_DATA_PATH = os.path.join(ROOT_PATH, 'data/raw/heart_cleveland_upload.csv')

COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                   'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']

NUMERICAL = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
CATEGORICAL = ['cp',  'restecg', 'slope', 'thal', 'sex', 'fbs', 'exang', 'condition']

ORIG_DATA = pd.read_csv(ORIGINAL_DATA_PATH)


def make_fake_numerical(num_rows: int) -> pd.DataFrame:
    fake_data = pd.DataFrame(0, index=range(num_rows), columns=NUMERICAL)
    for col in NUMERICAL:
        col_mean = ORIG_DATA[col].mean()
        col_std = ORIG_DATA[col].var()
        fake_data[col] = np.random.normal(col_mean, col_std, num_rows)
    return fake_data


def make_fake_categorical(num_rows: int) -> pd.DataFrame:
    fake_data = pd.DataFrame(0, index=range(num_rows), columns=CATEGORICAL)
    for col in CATEGORICAL:
        unique, counts = np.unique(ORIG_DATA[col].to_numpy(), return_counts=True)
        probs = counts / len(ORIG_DATA)
        fake_data[col] = np.random.choice(a=unique, size=num_rows, p=probs)
    return fake_data


def make_fake_dataset(num_rows: int, save_path: str) -> str:
    fake_data = pd.DataFrame(index=range(num_rows), columns=COLUMNS)
    fake_data[NUMERICAL] = make_fake_numerical(num_rows)
    fake_data[CATEGORICAL] = make_fake_categorical(num_rows)
    full_path = os.path.join(save_path, f"fake_{num_rows}.csv")
    fake_data.to_csv(full_path, index=False)
    return full_path
