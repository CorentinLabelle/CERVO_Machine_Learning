import sys
import numpy as np
from enum import Enum
from typing import List

sys.path.append('.')

from util import read_mat


class ChannelType(Enum):
    MEG = "MEG"
    EEG = "EEG"
    SEEG = "SEEG"

    def compare_with_string(self, string: str) -> bool:
        string = string.upper()
        return self.value == string

    @classmethod
    def from_string(cls, string: str) -> "ChannelType":
        string = string.upper()
        if string == ChannelType.EEG.value:
            return ChannelType.EEG
        elif string == ChannelType.MEG.value:
            return ChannelType.MEG
        elif string == ChannelType.SEEG.value:
            return ChannelType.SEEG
        else:
            raise Exception(f"Invalid string in input: {string}")


class BstChannelFile:
    """
    https://neuroimage.usc.edu/brainstorm/Tutorials/ChannelFile#Edit_the_channel_file
    """

    def __init__(self, channel_file_path: str):
        self.file_path: str = channel_file_path

        self.names: List[str] = []
        self.types: List[str] = []
        self.groups: List[str] = []
        self.locations: List[np.ndarray] = []

        self.__parse_channel_file__()

        self.channel_type: ChannelType = self.__detect_channel_type__()

        self.python_channel_indexes: List[int] = self.__get_channel_indexes__()
        self.matlab_channel_indexes: List[int] = [index + 1 for index in self.python_channel_indexes]

    def __parse_channel_file__(self) -> None:
        raw_file = read_mat.with_scipy(self.file_path, with_key=False)
        raw_channels = raw_file['Channel'].squeeze()

        for raw_channel in raw_channels:
            self.names.append(raw_channel.Name)
            self.types.append(raw_channel.Type)

            group = raw_channel.Group if len(raw_channel.Group) != 0 else ''
            self.groups.append(group)

            self.locations.append(raw_channel.Loc)

    def __detect_channel_type__(self) -> ChannelType:
        isEEG = "EEG" in self.types
        isMEG = "MEG" in self.types
        isSEEG = "SEEG" in self.types

        types = [isEEG, isMEG, isSEEG]

        if sum(types) > 1:
            raise Exception("Multiple types detected (EEG, MEG, SEEG)")
        elif sum(types) == 0:
            raise Exception("No type detected")
        else:
            if isEEG:
                return ChannelType.EEG
            elif isMEG:
                return ChannelType.MEG
            else:
                return ChannelType.SEEG

    def __get_channel_indexes__(self) -> List[int]:
        channel_indexes = list()
        for iChannel, channel_type in enumerate(self.types):
            if self.channel_type.compare_with_string(channel_type):
                channel_indexes.append(iChannel)
        return channel_indexes

    def get_matlab_channel_indexes(self) -> List[int]:
        return self.matlab_channel_indexes

    def get_python_channel_indexes(self) -> List[int]:
        return self.python_channel_indexes

    def __str__(self) -> str:
        return f"File: {self.file_path}\n" \
               f"Channel type: {self.channel_type.value}\n" \
               f"Number of channels: {len(self.names)}"


if __name__ == "__main__":
    channel_path = \
        "/mnt/3b5a15cf-20ff-4840-8d84-ddbd428344e9/ALAB1/rg/bs_db/Loca_intra_AUDI/data/HEJ_Subject01/1/channel.mat"
    channel_file = BstChannelFile(channel_path)
    print(channel_file)
