import pydantic as pydantic
from pydantic import typing


class BaseModelForbidExtraAttrs(pydantic.BaseModel):
    class Config:
        extra = "forbid"


class DataParams(BaseModelForbidExtraAttrs):
    DB_dir: str
    use_centroids: bool = False
    min_hh_size: int
    max_hh_size: int
    hh_num: int
    guests_per_hh: int
    household_size: int
    enrollment_utt: int
    evaluation_utt: int
    random_batch: bool = False
    saved_data: str = None
    path_to_households: str = None


class ScoringModelParams(BaseModelForbidExtraAttrs):
    dropout_type: str = 'original'
    input_dropout_rate: float
    adaptation_input_features: int
    adaptation_output_features: int
    use_bias: bool


class TrainingParams(BaseModelForbidExtraAttrs):
    batch_size: int
    epoch_num: int
    learning_rate: float
    num_workers: int


class ContinueFromParams(BaseModelForbidExtraAttrs):
    model_path: str


class Params(BaseModelForbidExtraAttrs):
    train_data: DataParams = None
    eval_data: DataParams
    scoring_model: ScoringModelParams = None
    training: TrainingParams = None
    continue_from: typing.Optional[ContinueFromParams] = None
    only_validate: bool = False

