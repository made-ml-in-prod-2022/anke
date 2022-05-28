import yaml

from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .data_params import SplitParams, PreproParams
from .model_params import TrainParams

from omegaconf import DictConfig


@dataclass()
class RunPipelineParams:
    input_dataset_path: str
    numpy_random_seed: int
    split_params: SplitParams
    prepro_params: PreproParams
    model_params: TrainParams


RunPipelineParamsSchema = class_schema(RunPipelineParams)


def read_run_pipeline_params(cfg: DictConfig) -> RunPipelineParams:
    schema = RunPipelineParamsSchema()
    return schema.load(cfg)
