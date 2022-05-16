import logging
import sys

from data import Dataset
from params import read_run_pipeline_params
from models import Model
import numpy as np

from omegaconf import DictConfig
import hydra

CONFIG_PATH = 'configs/config.yaml'

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@hydra.main(config_path='configs', config_name='config')
def run_pipeline(cfg: DictConfig):
    run_pipeline_params = read_run_pipeline_params(cfg)

    np.random.seed(run_pipeline_params.numpy_random_seed)
    logger.info("Start train pipeline")

    logger.info("Start reading and preprocessing data")
    dataset_path = hydra.utils.to_absolute_path(run_pipeline_params.input_dataset_path)
    dataset = Dataset(dataset_path)
    logger.info(f"Data.shape is {dataset.data.shape}")

    preprocessing_params = run_pipeline_params.prepro_params

    if preprocessing_params.transformer_preprocess:
        dataset.preprocess_with_transformer(preprocessing_params)
        logger.info(f"Data was preprocessed with ColumnTransformer")
    else:
        dataset.preprocess(preprocessing_params)

    if preprocessing_params.pca:
        dataset.apply_pca(preprocessing_params)

    logger.info(f"Preprocessed data.shape is {dataset.data.shape}")

    split_params = run_pipeline_params.split_params
    if split_params.split:
        X_train, X_test, y_train, y_test = dataset.split(split_params)
        logger.info(f"X_train.shape, y_train.shape are {X_train.shape, y_train.shape}")
        logger.info(f"X_test.shape, y_test.shape are {X_test.shape, y_test.shape}")
        logger.info(f"X_train looks like this:\n{X_train.head()}")
    else:
        X_data, y_data = dataset.get_data_and_target(split_params)
        logger.info(f"Features look like this:\n{X_data.head()}")

    logger.info("Preprocessing finished")

    model_params = run_pipeline_params.model_params
    model_params.ckpt_path = hydra.utils.to_absolute_path(model_params.ckpt_path)
    model_params.preds_path = hydra.utils.to_absolute_path(model_params.preds_path)
    model_params.eval_path = hydra.utils.to_absolute_path(model_params.eval_path)
    model = Model(model_params)
    logger.info(f"Created model {model.name}")

    if model_params.train:
        logger.info("Start training")
        if split_params.split:
            model.train(X_train, y_train)
        else:
            model.train(X_data, y_data)

        if model_params.ckpt_path is not None:
            model_path = model.serialize_estimator(model_params.ckpt_path)
            logger.info(f"Model serialized to: {model_path}")

        logger.info("Training finished")

    if model_params.test:
        logger.info("Start inference")
        if split_params.split:
            X_data = X_test
            y_data = y_test
        preds = model.predict(X_data)
        if model_params.preds_path is not None:
            preds_path = model.dump_preds(preds, model_params.preds_path)
            logger.info(f"Predictions saved to: {preds_path}")
        if model_params.evaluate_test:
            eval_results = model.eval(preds, y_data)
            logger.info(f"Evaluation results: {eval_results.items()}")
            if model_params.eval_path is not None:
                eval_path = model.dump_eval(eval_results, model_params.eval_path)
                logger.info(f"Evaluation results saved to: {eval_path}")


if __name__ == "__main__":
    run_pipeline()
