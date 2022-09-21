import pydantic as pydantic



class BaseModelForbidExtraAttrs(pydantic.BaseModel):
    class Config:
        extra = "forbid"

class DataParams(BaseModelForbidExtraAttrs):
    DB_dir: str
    min_hh_size: int
    max_hh_size: int
    hh_num: int
    guests_per_hh: int
    household_size: int
    enrollment_utt: int
    evaluation_utt: int
    random_batch: pydantic.StrictBool


class ScoringModelParams(BaseModelForbidExtraAttrs):
    input_dropout_rate: float
    adaptation_input_features: int
    adaptation_output_features: int
    use_bias: pydantic.StrictBool


class Params(BaseModelForbidExtraAttrs):
    data: DataParams
    scoring_model: ScoringModelParams
