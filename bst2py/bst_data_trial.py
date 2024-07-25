import numpy as np
import os
from typing import List

import util.util
from util import read_mat


class BstDataTrial:

    def __init__(self, data_trial_path: str):
        self.data_trial_path: str = data_trial_path
        self.class_name: str = ""
        self.class_index: int = -1

        self.bad_trial: bool = False
        self.channel_indexes: List[int] = []

    def get_filepath(self) -> str:
        return self.data_trial_path

    def get_filename(self) -> str:
        return os.path.basename(self.get_filepath())

    def set_as_bad_trial(self):
        self.bad_trial = True

    def is_bad_trial(self) -> bool:
        return self.bad_trial

    def get_raw_data(self, keep_all_channel: bool = False) -> np.ndarray:
        raw_data = read_mat.with_scipy(self.data_trial_path, with_key=False)['F']
        if keep_all_channel:
            return raw_data
        else:
            channel_indexes = self.get_channel_indexes()
            if len(channel_indexes) == 0:
                return raw_data
            else:
                return raw_data[self.get_channel_indexes()]

    def get_times(self) -> np.ndarray:
        return read_mat.with_scipy(self.data_trial_path, with_key=False)['Time'].squeeze()

    def get_channel_flags(self) -> np.ndarray:
        return read_mat.with_scipy(self.data_trial_path, with_key=False)['ChannelFlag'].squeeze()

    def get_events(self) -> dict:
        raw_data_trial = read_mat.with_scipy(self.data_trial_path, with_key=False)
        return __parse_events__(raw_data_trial['Events'])

    def set_channel_indexes(self, channel_indexes: List[int]) -> None:
        self.channel_indexes = channel_indexes

    def get_channel_indexes(self) -> List[int]:
        return self.channel_indexes

    def set_class_name(self, class_name: str) -> None:
        self.class_name = class_name

    def get_class_name(self) -> str:
        return self.class_name

    def extract_class_name_from_filename(self) -> str:
        filename = self.get_filename()
        first_underscore = filename.find('_')
        last_underscore = filename.rfind('_')
        class_label = filename[first_underscore + 1:last_underscore]
        return class_label

    def __str__(self) -> str:
        return f"File: {self.data_trial_path}\n"


def __parse_events__(raw_events: np.ndarray) -> dict:
    raw_events = raw_events.squeeze()
    fields = ['label', 'epochs', 'times']
    events = {field: [] for field in fields}

    for raw_event in raw_events:
        events['label'].append(raw_event[0][0])
        events['epochs'].append(raw_event[2])
        events['times'].append(raw_event[3])
    return events


if __name__ == "__main__":
    path = \
        "/mnt/3b5a15cf-20ff-4840-8d84-ddbd428344e9/ALAB1/rg/bs_db/Loca_intra_AUDI/data/HEJ_Subject01/1/data_1_trial001.mat"
    data_trial = BstDataTrial(path)
    print(data_trial)
