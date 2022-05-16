from typing import List
from dataclasses import dataclass, field


@dataclass()
class LoadParams:
    input_dataset_path: str = field(default="data/raw/heart_cleveland_upload.csv")


@dataclass()
class PreproParams:
    feature_names: List[str]
    correlated_features: List[str]
    one_hot_features: List[str]
    norm_features: List[str]
    target_column: str

    transformer_preprocess: bool = field(default=False)
    delete_correlated: bool = field(default=True)
    norm: bool = field(default=True)
    pca: bool = field(default=True)
    pca_n_comp: int = field(default=10)


@dataclass()
class SplitParams:
    split: bool = field(default=True)
    test_size: float = field(default=0.25)
    random_state: int = field(default=1234)
    shuffle: bool = field(default=False)
    target_column: str = field(default='condition')
