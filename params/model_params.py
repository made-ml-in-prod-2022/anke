from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class TrainParams:
    name: Optional[str] = None
    # linreg params
    penalty: Optional[str] = 'l2'
    solver: Optional[str] = 'liblinear'
    reg_param: Optional[float] = 1

    # Trees&RandomForest params
    max_depth: Optional[int] = 10
    min_samples_leaf: Optional[int] = 1
    max_features: Optional[str] = 'sqrt'

    ckpt_path: Optional[str] = None
    preds_path: Optional[str] = None
    eval_path: Optional[str] = None
    eval_from_path: Optional[bool] = False

    model_type: str = field(default="RandomForest")
    random_state: int = field(default=1234)

    train: bool = field(default=True)
    test: bool = field(default=True)
    evaluate_test: bool = field(default=True)

    def __post_init__(self):
        if self.name is None:
            self.name = self.model_type


