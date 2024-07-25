import configparser
import os
from dataclasses import dataclass
from typing import List

import numpy as np

from models.linear_models.linear_models import PipelineFactory, BaseLinearPipeline
from util import read_mat, abstract_input


@dataclass
class TrainingInput(abstract_input.AbstractInput):
    output_folder: str = None

    protocol_directory: str = None
    protocol_name: str = "default_protocol_name"
    subjects: List[str] = None
    condition: List[str] = None

    class_folders: List[tuple] = None
    class_folders_test: List[tuple] = None
    start_sample: int = 0
    end_sample: int = None
    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = None

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

    sampling_frequency: int = None
    high_freq: float = None
    low_freq: float = None

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

        if not isinstance(self.subjects, list):
            raise Exception(f"'subjects' has to be a list.")

        if not (self.class_folders is None or isinstance(self.class_folders, list)):
            raise Exception(f"'class_folders' has to be a list or None.")

        if not (self.class_folders_test is None or isinstance(self.class_folders_test, list)):
            raise Exception("'class_folders_test' has to be a list or None.")

        if not (self.start_sample is None or isinstance(self.start_sample, int)):
            raise Exception("Start sample has to be an integer or None.")

        if not (self.end_sample is None or isinstance(self.end_sample, int)):
            raise Exception("End sample has to be an integer or None.")

        possible_pre_processing = ("AVERAGE", "VECTORIZE")
        if not (self.pre_processing.upper() in possible_pre_processing):
            raise Exception(f"'pre_processing' has to be one of these values: {possible_pre_processing}")

        if self.condition is None:
            self.condition = [""]

    def create_pipelines(self) -> List[BaseLinearPipeline]:
        pipelines = []
        if self.run_ridge:
            pipelines.append(PipelineFactory.create(pipeline_type="RIDGE", pre_processing=self.pre_processing))
        elif self.run_logistic:
            pipelines.append(PipelineFactory.create(pipeline_type="LOGISTIC", pre_processing=self.pre_processing))
        elif self.run_perceptron:
            pipelines.append(PipelineFactory.create(pipeline_type="PERCEPTRON", pre_processing=self.pre_processing))
        elif self.run_svc:
            pipelines.append(PipelineFactory.create(pipeline_type="SVC", pre_processing=self.pre_processing))
        return pipelines

    def condition_to_string(self) -> str:
        if self.condition is None:
            return ""
        elif len(self.condition) == 1:
            return self.condition[0]
        elif len(self.condition) == 2:
            return f"{self.condition[0]}_on_{self.condition[1]}"
        else:
            raise Exception(f"Invalid number of conditions: {self.condition}")


if __name__ == "__main__":
    cfg_file = "/training/training_input_template/as_config.ini"
    mat_file = "../ici.mat"
    ti_1 = TrainingInput.read(cfg_file)
    ti_2 = TrainingInput.read(mat_file)
    print("done.")
