import os.path
from typing import List
import glob

from bst2py import load_x_y
from util import util


class BstProtocol:
    """
    https://neuroimage.usc.edu/brainstorm/Tutorials/Database?highlight=%28database%29#On_the_hard_drive
    """

    def __init__(self, directory: str,
                 subject_patterns_to_keep: List[str] = None, subject_patterns_to_remove: List[str] = None,
                 train_studies: List[str] = None, test_studies: List[str] = None):
        self.directory: str = directory

        self.subjects = self.__get_subjects__(subject_patterns_to_keep, subject_patterns_to_remove)
        if len(self.subjects) == 0:
            self.subjects = [""]

        self.train_studies: List[str] = train_studies
        self.test_studies: List[str] = test_studies

    def get_data_trials(self) -> tuple:
        train_data_trials = []
        test_data_trials = []

        for subject in self.subjects:
            subject_folder = util.join_path(self.directory, subject)

            train_data_trials.append(
                load_x_y.load_data_trials(subject_folder, class_folder_names=self.train_studies)
            )

            test_data_trial = None
            if self.test_studies is not None:
                test_data_trial = load_x_y.load_data_trials(subject_folder, class_folder_names=self.test_studies)
            test_data_trials.append(test_data_trial)

        return self.subjects, train_data_trials, test_data_trials

    def __get_subjects__(self, subject_patterns_to_keep: List[str], subject_patterns_to_remove: List[str]) -> List[str]:
        subjects_to_keep = []
        if subject_patterns_to_keep is not None:
            for subject_pattern in subject_patterns_to_keep:
                subjects_to_keep.extend(glob.glob(util.join_path(self.directory, subject_pattern)))

        subjects_to_remove = []
        if subject_patterns_to_remove is not None:
            for subject_pattern in subject_patterns_to_remove:
                subjects_to_remove.extend(glob.glob(util.join_path(self.directory, subject_pattern)))

        subjects = list(set(subjects_to_keep) - set(subjects_to_remove))
        subjects = [os.path.basename(subject) for subject in subjects]
        return util.sort_list_in_alpha_num_order(subjects)
