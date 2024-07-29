import sys
import os
import warnings
import numpy as np
from typing import List

sys.path.append('.')

from util import process_data, util
from bst2py import bst_channel_file, bst_data_trial, bst_study


def load_x_y_labels(imported_data_folder: str, class_folder_names: list = None) -> tuple:
    data_trials = load_data_trials(imported_data_folder, class_folder_names)
    return data_trials_to_x_y_labels(data_trials)


def data_trials_to_x_y_labels(data_trials: List[bst_data_trial.BstDataTrial]) -> tuple:

    # Get class labels
    class_labels = [data_trial.get_class_name() for data_trial in data_trials]
    class_labels = list(set(class_labels))

    # Mapping of class labels to class indexes
    class_labels = util.sort_list_in_alpha_num_order(class_labels)

    # Loop over classes
    x = []
    y = []
    for data_trial in data_trials:
        if data_trial.is_bad_trial():
            continue
        x.append(data_trial.get_raw_data(keep_all_channel=False))
        class_index = class_labels.index(data_trial.get_class_name())
        y.append(class_index)

    return np.array(x), np.array(y), class_labels


def load_data_trials(imported_data_folder: str, class_folder_names: list = None) -> List[bst_data_trial.BstDataTrial]:

    # If None --> Detecting if unique folder or separated
    if class_folder_names is None:
        data_trials_organized_in_separate_folders = \
            util.directory_contains_subdirectory(imported_data_folder)
        if data_trials_organized_in_separate_folders:
            class_folder_names = util.list_directory_content(imported_data_folder)

    elif len(class_folder_names) == 1:
        data_trials_organized_in_separate_folders = False
        imported_data_folder = util.join_path(imported_data_folder, class_folder_names[0])

    # If list --> Separated folders
    elif isinstance(class_folder_names, list):
        data_trials_organized_in_separate_folders = True

    # Raise error
    else:
        raise Exception(f"Unsupported type: {type(class_folder_names)}")

    # Load files
    if data_trials_organized_in_separate_folders:
        data_trials = __load_files_from_separate_folders__(imported_data_folder, class_folder_names)
    else:
        data_trials = __load_files_from_unique_folder__(imported_data_folder)

    return data_trials


def __load_files_from_separate_folders__(
        imported_data_folder: str, classes_folders: list) -> List[bst_data_trial.BstDataTrial]:
    """
    │─── <class_0>
    │    │─── brainstormstudy.mat
    │    │─── channel*.mat
    │    │─── data_<class_0>_trialXXX.mat
    │    │─── data_<class_0>_trialXXX.mat
    │    │─── ...
    │
    │─── <class_1>
    │    │─── brainstormstudy.mat
    │    │─── channel*.mat
    │    │─── data_<class_1>_trialXXX.mat
    │    │─── data_<class_1>_trialXXX.mat
    │    │─── ...
    │
    │─── ...

    :returns
        data trial paths per class: list of <nb_classes> lists, path to data trials.
        class labels: list of <nb_classes>.
        channel indexes: list, a single list for all classes.
        bad trials: list, a single list for all classes.
    """

    data_trials = []

    # Loop over class folder names
    for class_folders in classes_folders:

        # Get class label
        if isinstance(class_folders, tuple):
            class_label = f'{class_folders[0]}_merged'

        elif isinstance(class_folders, str):
            class_label = class_folders

            # One folder for that class, convert to tuple
            class_folders = class_folders,

        else:
            raise Exception(f'Unsupported type: {type(class_folders)}')

        # Loop over folders for that class
        for class_folder in class_folders:
            class_folder_path = util.join_path(imported_data_folder, class_folder)

            # Load files (as a unique folder for one class)
            current_folder_data_trials = __load_files_from_unique_folder__(
                class_folder_path, class_name_for_all=class_label)

            data_trials.extend(current_folder_data_trials)

    return data_trials


def __load_files_from_unique_folder__(imported_data_folder: str, class_name_for_all: str = None)\
        -> List[bst_data_trial.BstDataTrial]:
    """
    Get all the class labels from a Brainstorm folder organized as such:
    │─── <imported_data_folder>
    │    │─── brainstormstudy.mat
    │    │─── data_<class_1>_trialXXX.mat
    │    │─── data_<class_1>_trialXXX.mat
    │    │─── ...
    │    │─── data_<class_2>_trialXXX.mat
    │    │─── data_<class_2>_trialXXX.mat
    │    │─── ...
    │    │─── data_<class_3>_trialXXX.mat
    │    │─── data_<class_3>_trialXXX.mat
    │    │─── ...
    """

    # Get bad trials
    brainstorm_study_path = util.find_with_patterns(imported_data_folder, "brainstormstudy.mat")
    if len(brainstorm_study_path) == 0:
        warnings.warn("No brainstormstudy.mat found, cannot remove bad trials.")
        bad_trials = []
    elif len(brainstorm_study_path) > 1:
        raise Exception(f"Multiple brainstormstudy.mat file found in folder: {str(imported_data_folder)}")
    else:
        brainstorm_study = bst_study.BstStudy(str(brainstorm_study_path[0]))
        bad_trials = brainstorm_study.get_bad_trials()

    # Get channel indexes
    channel_file_path = util.find_with_patterns(imported_data_folder, "channel*.mat")
    if len(channel_file_path) == 0:
        warnings.warn("No channel*.mat found, cannot extract channel indexes.")
        channel_indexes = []
    elif len(channel_file_path) > 1:
        raise Exception(f"Multiple channel*.mat file found in folder: {str(imported_data_folder)}")
    else:
        channel_file = bst_channel_file.BstChannelFile(str(channel_file_path[0]))
        channel_indexes = channel_file.get_python_channel_indexes()

    # Get data trial files
    data_trial_paths = util.find_with_patterns(imported_data_folder, "data_*.mat")

    # Loop over data trials to get class labels
    data_trials = []
    for data_trial_path in data_trial_paths:
        data_trial = bst_data_trial.BstDataTrial(str(data_trial_path))

        data_trial.set_channel_indexes(channel_indexes)

        if data_trial.get_filename() in bad_trials:
            data_trial.set_as_bad_trial()

        if class_name_for_all is None:
            class_name = data_trial.extract_class_name_from_filename()
        else:
            class_name = class_name_for_all
        data_trial.set_class_name(class_name)

        data_trials.append(data_trial)

    return data_trials


if __name__ == '__main__':

    # One condition, regrouped in one folder
    protocol = \
        '/mnt/3b5a15cf-20ff-4840-8d84-ddbd428344e9/ALAB1/corentin/projects/machine_learning/data/SEEG/art_replay/'
    subject = 'HEJ_Subject01/'
    condition = 'P1'
    class_folders = None

    data_directory = os.path.join(protocol, subject, condition)

    X, y = load_x_y_labels(data_directory, class_folder_names=class_folders)

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    process_data.print_distribution(y)
    print()

    # No condition
    protocol = '/mnt/3b5a15cf-20ff-4840-8d84-ddbd428344e9/ALAB1/corentin/projects/machine_learning/data/SEEG/loca_audi/'
    subject = 'HEJ_Subject01/'
    condition = ''
    class_folders = None

    data_directory = os.path.join(protocol, subject, condition)

    X, y = load_x_y_labels(data_directory, class_folder_names=class_folders)

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    process_data.print_distribution(y)
    print()

    # Train on one condition, test on another
    protocol = '/mnt/7012ad01-e4b3-459b-9944-8395b35698e6/ALAB/temp_shiva/'
    subject = ''
    condition = ('train', 'test')
    class_folders = None

    train_data_directory = os.path.join(protocol, subject, condition[0])
    X_train, y_train = load_x_y_labels(train_data_directory, class_folder_names=class_folders)

    test_data_directory = os.path.join(protocol, subject, condition[1])
    X_test, y_test = load_x_y_labels(test_data_directory, class_folder_names=class_folders)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    print()
