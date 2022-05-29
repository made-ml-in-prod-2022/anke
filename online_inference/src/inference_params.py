from typing import List
import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema


@dataclass()
class InferenceParams:
    # data_path: str
    model_path: str
    feature_names: List[str]
    correlated_features: List[str]
    one_hot_features: List[str]
    norm_features: List[str]


InferenceParamsSchema = class_schema(InferenceParams)


def read_inference_params(cfg_path) -> InferenceParams:
    schema = InferenceParamsSchema()
    with open(cfg_path, 'r') as f:
        return schema.load(yaml.safe_load(f))
