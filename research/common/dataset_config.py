from dataclasses import dataclass, field
from dataclass_type_validator import dataclass_validate
from enum import IntEnum, auto
from typing import Union, List, Optional


class DatasetTask(IntEnum):
    CLASSIFICATION = auto()
    PREDICTION = auto()


class PatientCohort(IntEnum):
    CN = auto()
    MCI = auto()
    Dementia = auto()
    sMCI = auto()
    pMCI = auto()

    def dx_to_cohort(dx):
        if dx == "CN":
            return PatientCohort.CN
        elif dx == "MCI":
            return PatientCohort.MCI
        elif dx == "Dementia":
            return PatientCohort.Dementia
        else:
            raise RuntimeError(f"Unknown cohort: {dx}")

    def is_classification(self) -> bool:
        return (
            self.value == PatientCohort.CN
            or self.value == PatientCohort.MCI
            or self.value == PatientCohort.Dementia
        )

    def is_prediction(self) -> bool:
        return self.value == PatientCohort.sMCI or self.value == PatientCohort.pMCI

    def get_task_type(self) -> DatasetTask:
        if self.is_classification():
            return DatasetTask.CLASSIFICATION
        elif self.is_prediction():
            return DatasetTask.PREDICTION
        else:
            raise RuntimeError(f"Unknown task type: {self.value}")

    def get_ordinal(self, cohorts: Optional[List[str]]):
        if len(cohorts) > 0:
            return cohorts.index(self.name)
        else:
            index_mapping = {
                PatientCohort.CN: 0,
                PatientCohort.MCI: 1,
                PatientCohort.Dementia: 2,
                PatientCohort.sMCI: 0,
                PatientCohort.pMCI: 1,
            }
            return index_mapping[self.value]


class DataMode(IntEnum):
    SCANS = auto()
    PATHS = auto()


# @dataclass_validate
@dataclass
class NestedCV:
    num_inner_fold: int
    num_outer_fold: int


# @dataclass_validate
@dataclass
class FlatCV:
    num_folds: int
    test_ratio: float


# @dataclass_validate
@dataclass
class BasicSplit:
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def sum(self):
        return self.train_ratio + self.val_ratio + self.test_ratio


@dataclass
class NoSplit:
    pass


SplitTypes = Union[NestedCV, FlatCV, BasicSplit, NoSplit]


# @dataclass_validate
@dataclass
class DatasetConfig:
    task: DatasetTask
    mode: DataMode
    split_type: SplitTypes

    scan_paths: str

    batch_size: int

    load_embeddings: bool
    num_seq_visits: int
    seq_visit_delta: int
    progression_window: int
    tolerance_lower: int
    tolerance_upper: int

    embedding_paths: Optional[str] = None
    cohorts: Optional[List[str]] = field(default_factory=list)
    ni_vars: Optional[List[str]] = field(default_factory=list)
