# HW1 репозиторий

## Структура:
📁 ml_project/
├─📁 configs/                                   # yaml конфиги для запуска
│ ├─📄 config.yaml
│ └─📄 config_2.yaml
├─📄 criterions.txt                               # самооценка
├─📁 data/                                      # оригинальный датасет, предобработка и код для подражания
│ ├─📄 Data.py
│ ├─📄 fake_data.py
│ ├─📁 raw/
│ │ └─📄 heart_cleveland_upload.csv
│ └─📄 __init__.py
├─📁 models/                                    # класс модели(train, predict, eval, save)
│ ├─📄 Train.py
│ └─📄 __init__.py
├─📁 notebook/                                  # jupyter notebook
│ └─📄 heart_stats.ipynb
├─📁 params/                                    # dataclasses 
│ ├─📄 data_params.py
│ ├─📄 model_params.py
│ ├─📄 train_pipeline_params.py
│ └─📄 __init__.py
├─📄 requirements.txt                             # зависимости
├─📄 run_pipeline.py                              # пайплайн обучения и теста
├─📁 saves/                                     # артефакты
│ ├─📁 chekpoints/
│ ├─📁 eval_results/
│ └─📁 preds/
└─📁 tests/                                     # тесты
  ├─📄 data_tests.py
  └─📄 model_tests.py

## Usage
**train, test and evaluate RandomForestClassifier, save model to saves/checkpoints, save predictions to saves/preds, save metrics to saves/eval_results:**
`python -m run_pipeline`

**train RandomForestClassifier, save model to saves/checkpoints:**
`python -m run_pipeline model_params.test=false`

**test and evaluate pickled model from** *path*, **save predictions to saves/preds, save metrics to saves/eval_results:**
`python -m run_pipeline model_params.train=false model_params.eval_from_path=true model_params.ckpt_path=<path>`
  
**Optional arguments:**
`model_params.name=<str>` - model, preds and metrics will be saved with this name in filename
`model_params.model_type=<RandomForestClassifier, DecisionTreeClassifier, LogisticRegression>` - choose model type 
`model_params.preds_path=<str>` - specify predictions path
`model_params.eval_path=<str>` - specify metric results path 
`prepro_params.pca=true model_params.pca_n_comp=<int>` - apply pca with number of components = pca_n_comp, default=10 

**run tests:**
`python -m pytest --cov data/ --cov models/  tests/data_tests.py tests/model_tests.py`
