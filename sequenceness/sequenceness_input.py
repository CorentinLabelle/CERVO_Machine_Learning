import configparser
import os
from dataclasses import dataclass
from typing import List

import numpy as np

from util import read_mat, abstract_input


@dataclass
class SequencenessInput(abstract_input.AbstractInput):
    output_folder: str = None

    data_folder: str = None

    label_folder: str = None
    label_index_starting_at_1: bool = True

    result_filepath: str = None
    pipeline_fields_in_result_file: List[str] = None

    compression_factors: List[int] = None

    start_sample: int = None
    end_sample: int = None

    def validate(self):
        pass


if __name__ == "__main__":
    cfg_file = "../empty_template_config.ini"
    mat_file = "../ici.mat"
    ti_1 = SequencenessInput.read(cfg_file)
    ti_2 = SequencenessInput.read(mat_file)
    print("done.")


