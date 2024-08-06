import os
from dataclasses import dataclass
from typing import List

from training.trainings import TrainingType
from models.linear_models.linear_models import PipelineFactory, BaseLinearPipeline
from configuration.abstract_configuration import AbstractConfiguration


@dataclass
class TrainingConfiguration(AbstractConfiguration):
    output_folder: str = None

    protocol_directory: str = None
    protocol_name: str = "default_protocol_name"
    subjects_to_keep: List[str] = None
    subjects_to_skip: List[str] = None
    train_studies: List[str] = None
    test_studies: List[str] = None

    start_sample: int = 0
    end_sample: int = None

    sampling_frequency: int = None
    high_freq: float = None
    low_freq: float = None

    run_ridge: bool = False
    run_logistic: bool = False
    run_perceptron: bool = False
    run_svc: bool = False
    pre_processing: str = None

    run_cross_validation: bool = False
    run_time_generalization: bool = False
    run_time_sampling: bool = False
    run_grid_search: bool = False
    nb_replications: int = 2
    test_size: float = 0.2
    n_splits: int = 5
    random_states: List[int] = None

    run_permutation: bool = False
    n_permutations: int = None
    permutation_n_splits: int = None

    def validate(self):

        if not isinstance(self.output_folder, str):
            raise Exception(f"'output_folder' has to be a string.")

        if not isinstance(self.protocol_directory, str):
            raise Exception(f"'protocol_directory' has to be a string.")

        if not os.path.isdir(self.protocol_directory):
            raise Exception(f"'protocol_directory' does not exists: {self.protocol_directory}.")

        if not (self.subjects_to_keep is None or isinstance(self.subjects_to_keep, list)):
            raise Exception(f"'subject_patterns_to_keep' has to be a list.")

        if not (self.train_studies is None or isinstance(self.train_studies, list)):
            raise Exception(f"'class_folders' has to be a list or None.")

        if not (self.test_studies is None or isinstance(self.test_studies, list)):
            raise Exception("'class_folders_test' has to be a list or None.")

        if not (self.start_sample is None or isinstance(self.start_sample, int)):
            raise Exception("Start sample has to be an integer or None.")

        if not (self.end_sample is None or isinstance(self.end_sample, int)):
            raise Exception("End sample has to be an integer or None.")

        if self.pre_processing is None:
            raise Exception(f"'pre_processing' cannot be empty.")

        possible_pre_processing = ("AVERAGE", "VECTORIZE")
        if not (self.pre_processing.upper() in possible_pre_processing):
            raise Exception(f"'pre_processing' has to be one of these values: {possible_pre_processing}")

    def create_pipelines(self) -> List[BaseLinearPipeline]:
        pipelines = []
        if self.run_ridge:
            pipelines.append(PipelineFactory.create(pipeline_type="RIDGE", pre_processing=self.pre_processing))
        if self.run_logistic:
            pipelines.append(PipelineFactory.create(pipeline_type="LOGISTIC", pre_processing=self.pre_processing))
        if self.run_perceptron:
            pipelines.append(PipelineFactory.create(pipeline_type="PERCEPTRON", pre_processing=self.pre_processing))
        if self.run_svc:
            pipelines.append(PipelineFactory.create(pipeline_type="SVC", pre_processing=self.pre_processing))
        return pipelines

    def get_training_types(self) -> List[TrainingType]:
        training_types = list()
        if self.run_cross_validation:
            training_types.append(TrainingType.CROSS_VALIDATION)
        if self.run_time_generalization:
            training_types.append(TrainingType.TIME_GENERALIZATION)
        return training_types


if __name__ == "__main__":
    cfg_file = "/training/training_config_template/as_training.ini"
    mat_file = "../ici.mat"
    ti_1 = TrainingConfiguration.read(cfg_file)
    ti_2 = TrainingConfiguration.read(mat_file)
    print("done.")
