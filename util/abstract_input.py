import configparser
import os
from dataclasses import dataclass
from typing import List
from abc import abstractmethod, ABC

import numpy as np

from util import read_mat


@dataclass
class AbstractInput(ABC):

    @classmethod
    def read(cls, file_path: str):
        extension = os.path.splitext(file_path)[-1]
        if extension in (".ini", ".cfg"):
            return cls.from_cfg_file(file_path)
        elif extension == ".mat":
            return cls.from_mat_file(file_path)
        else:
            raise Exception(f"Invalid file extension: {extension}")

    @classmethod
    def from_dico(cls, dico: dict):
        training_input = cls()

        first_level_keys = list(dico.keys())
        for first_level_key in first_level_keys:
            second_level_keys = list(dico[first_level_key].keys())
            for second_level_key in second_level_keys:
                value = dico[first_level_key][second_level_key]

                if isinstance(value, np.ndarray):
                    value = None

                if value is None:
                    continue

                value_type = training_input.__annotations__[second_level_key]

                if value_type == List[str]:
                    value = value.strip().replace(" ", "").split(",")
                elif value_type == List[tuple]:
                    value = value.strip().replace(" ", "").split(",")
                    value = [tuple(v.replace(" ", "").split("-")) for v in value]
                elif value_type == List[int]:
                    value = value.strip().replace(" ", "").split(",")
                    value = [int(v.strip()) for v in value]
                elif value_type == int:
                    value = int(value)
                elif value_type == float:
                    value = float(value)
                elif value_type == bool:
                    value = bool(int(value))

                training_input.__setattr__(second_level_key, value)

        training_input.validate()
        return training_input

    @classmethod
    def from_cfg_file(cls, cfg_path: str):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(cfg_path)
        return cls.from_dico(config)

    @classmethod
    def from_mat_file(cls, mat_filepath: str):
        mat_file = read_mat.read_mat(mat_filepath, with_key=False)
        mat_file_filtered = {key: value for key, value in mat_file.items() if not key.startswith("__")}
        return cls.from_dico(mat_file_filtered)

    @abstractmethod
    def validate(self):
        pass
