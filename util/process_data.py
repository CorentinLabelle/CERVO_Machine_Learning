import numpy as np
import matplotlib.pyplot as plt
from typing import List

from sklearn.model_selection import train_test_split
import mne
from scipy import signal


def train_validation_test_split(X, y, validation_ratio, test_ratio,
                                random_state=None, shuffle: bool = True) -> tuple:

    X_train, X_validation, y_train, y_validation = \
        train_test_split(X, y,
                         test_size=validation_ratio,
                         random_state=random_state,
                         stratify=y, shuffle=shuffle)

    if test_ratio == 0:
        return X_train, X_validation, y_train, y_validation

    else:
        new_test_ratio = test_ratio / (1 - validation_ratio)
        X_train, X_test, y_train, y_test = \
            train_test_split(X_train, y_train,
                             test_size=new_test_ratio,
                             random_state=random_state,
                             stratify=y_train, shuffle=shuffle)

        return X_train, X_validation, X_test, y_train, y_validation, y_test


def slice_array(array: np.ndarray, axis: int, index_1: int = None, index_2: int = None) -> np.ndarray:
    slice_tuple = [slice(None)] * array.ndim
    slice_tuple[axis] = slice(index_1, index_2)
    return array[tuple(slice_tuple)]


def compress(array: np.ndarray, axis: int, sequence_length: int, time_compression_factor: int,
             end_frame: int = None) -> np.ndarray:
    if end_frame is None:
        end_frame = array.shape[axis]

    start_frame = end_frame - (sequence_length * time_compression_factor)
    if start_frame < 0:
        return np.ndarray([])

    array_sliced = slice_array(array, axis=axis, index_1=start_frame, index_2=end_frame)
    array_compressed = signal.resample(array_sliced, num=sequence_length, axis=axis)
    return array_compressed


def convert_epoch_to_timestamp_class_sample(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    For each epoch, we take each timestamp vector (a vector with a single value for each channel) and assign it the epoch's class.
    It creates a new matrix of shape (e*t, c, 1)
    :param x: (e, c, t)
    :param y: (e)
    :return: (e*t, c, 1)
    """
    new_x = list()
    new_y = list()

    nb_epochs = x.shape[0]
    nb_timestamps = x.shape[2]
    for i in range(nb_epochs):
        current_x = x[i, :, :]
        for j in range(nb_timestamps):
            new_y.append(y[i])
            new_x.append(current_x[:, j])

    new_x = np.array(new_x)
    new_x = np.expand_dims(new_x, axis=2)
    return new_x, np.array(new_y)


def band_pass_filter(x: np.ndarray, sampling_frequency: float,
                     low_freq: float = None, high_freq: float = None, verbose=False):
    return mne.filter.filter_data(x, sampling_frequency, low_freq, high_freq, method='iir', verbose=verbose)


def print_distribution(y) -> None:
    classes, count = np.unique(y, return_counts=True)
    distribution = count / len(y)

    print('Distribution')
    for iClass in range(len(classes)):
        print(f'{classes[iClass]}: {count[iClass]} - {distribution[iClass]:.2f}')
    print(f'Total: {len(y)}')


def graph_distribution(y_train: np.ndarray, y_test: np.ndarray = None):

    class_indexes = np.unique(np.concatenate((y_train, y_test)))

    train_classes, train_count = np.unique(y_train, return_counts=True)
    test_classes, test_count = np.unique(y_test, return_counts=True)

    fig = plt.figure()

    # Plot training
    train_bars = plt.bar(train_classes, train_count, label='Training', color='r', alpha=0.2)
    plt.bar_label(train_bars, label_type='center', fmt='%.0f')

    # Plot testing
    test_bars = plt.bar(test_classes, test_count, bottom=train_count, label='Testing', color='b', alpha=0.2)
    plt.bar_label(test_bars, label_type='center', fmt='%.0f')

    plt.bar_label(test_bars, labels=train_count+test_count, label_type='edge', fmt='%.0f')

    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.legend(loc='lower right')
    plt.title("Class distribution")
    plt.xticks(ticks=[i for i in range(len(class_indexes))], rotation=0)

    plt.title("Classes Distribution")
    plt.tight_layout()

    return fig


def create_class_label_index_mapping_file(class_labels: List[str], filepath: str):
    f = open(filepath, "w")
    for index, label in enumerate(class_labels):
        f.write(f"{index},{label}\n")


if __name__ == '__main__':

    if 0:
        fs = 1000
        T = 1
        nsamples = T * fs
        t = np.arange(0, nsamples) / fs
        x = 0.01 * np.sin(2 * np.pi * 20 * t)
        x += 0.01 * np.cos(2 * np.pi * 60 * t + 0.1)
        #x += 0.03 * np.cos(2 * np.pi * 2000 * t)

        x_f = band_pass_filter(x, 512, high_freq=30)

        plt.plot(x, label='raw')
        plt.plot(x_f, label='filtered')
        plt.legend()
        plt.show()
        print('Done.')

    filtered = '/mnt/3b5a15cf-20ff-4840-8d84-ddbd428344e9/ALAB1/corentin/projects/EEGNet_online_tool/docker_image/container_input/Face13_S01_S02_output/240319_104805/bst_db/data/sub-s02/sub-s02_task-faceFO_eeg_low/data_block001.mat'
    raw = '/mnt/7012ad01-e4b3-459b-9944-8395b35698e6/ALAB/temp_shiva/SEQ_MEG/All_M_WM1/data_trial_1.mat'

    from bst2py import data_trial_2py

    a = data_trial_2py.parse_data_trial(raw)
    raw_data = a['data']

    a = data_trial_2py.parse_data_trial(filtered)
    filtered_data = a['data']

    plt.plot(raw_data[0, :1000], label='raw')
    plt.plot(filtered_data[0, :1000], label='filtered')
    plt.legend()
    plt.draw()

    custom_filtered = band_pass_filter(raw_data, 256, high_freq=15)

    plt.figure()
    plt.plot(raw_data[0, :1000], label='raw')
    plt.plot(custom_filtered[0, :1000], label='custom filtered')
    plt.legend()
    plt.draw()
    plt.show()
