import json.decoder

import numpy as np
import requests
from test_data import make_fake_test_dataset

if __name__ == '__main__':

    num_rows = np.random.choice(range(1, 30))
    data = make_fake_test_dataset(num_rows)

    test_data = []
    for i in data.index:
        test_data.append([x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()])

    response = requests.get(
        'http://127.0.0.1:8000/predict/',
        json={'test_data': test_data},
    )

    if (
            response.status_code != 204 and
            response.headers["content-type"].strip().startswith("application/json")
    ):
        try:
            print(response.json())
        except json.decoder.JSONDecodeError:
            print('smth went wrong with data')

    print(response.status_code)
