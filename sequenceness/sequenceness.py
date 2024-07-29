import sys
import itertools
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from tqdm import tqdm

from bst2py.bst_data_trial import BstDataTrial

sys.path.append('../models/linear_models')

from util import read_pkl, read_mat, save_mat, save_pkl, process_data
from sequenceness.sequenceness_output import SequencenessOutput
from models.linear_models.linear_models import BaseLinearPipeline


def time_sequenceness(
        trained_pipeline: BaseLinearPipeline,
        data_trials: List[BstDataTrial], y: np.ndarray,
        compression_factors: List[int] = None,
        frame_1: int = None, frame_2: int = None, axis: int = 2):

    if compression_factors is None:
        # compression_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 80, 100]
        compression_factors = [1, 5]
    elif isinstance(compression_factors, int):
        compression_factors = [compression_factors]

    sequence_length = y.shape[1]
    possible_sequences = trained_pipeline.get_possible_sequences(sequence_length)

    data_trial_paths = [data_trial.get_filepath() for data_trial in data_trials]
    X = np.array([data_trial.get_raw_data(keep_all_channel=False) for data_trial in data_trials])

    # Slice times
    if frame_1 is not None or frame_2 is not None:
        X = process_data.slice_array(X, axis=2, index_1=frame_1, index_2=frame_2)

    outputs = []
    for compression_factor in tqdm(compression_factors):

        output = SequencenessOutput(
           possible_sequences=possible_sequences,
           compression_factor=compression_factor,
           expected_sequences=y,
           frames=(frame_1, frame_2),
           data_trial_filepaths=data_trial_paths
        )

        end_frame = X.shape[axis]
        while True:

            X_compressed = process_data.compress(
                X, axis=axis, sequence_length=sequence_length,
                time_compression_factor=compression_factor,
                end_frame=end_frame)

            if X_compressed.ndim == 0:
                break

            predicted_sequence = trained_pipeline.predict_sequence(X_compressed)
            probabilities = trained_pipeline.predict_sequence_probabilities(X_compressed)

            output.add_sequence_prediction(
                frame_of_compression_factor=end_frame,
                predicted_sequences=predicted_sequence,
                probabilities=probabilities
            )

            end_frame -= 1

        outputs.append(output)

    return outputs


def pac_sequenceness(
        trained_pipeline: BaseLinearPipeline,
        data_trials: List[BstDataTrial], y: np.ndarray,
        compression_factors: List[int] = None,
        frame_1: int = None, frame_2: int = None, axis: int = 2):
    pass


def __predict_probabilities_of_all_possible_sequences__(probabilities):
    """
    Input
        probabilities: (nb_epochs, nb_timestamps, nb_classes)
    Return
        all_possible_combinations: (nb_combinations) = (nb_classes ^ sequence_length)
        combination_probabilities: (nb_combinations) = (nb_classes ^ sequence_length)
    """
    sequence_length = probabilities.shape[1]
    nb_classes = probabilities.shape[2]

    all_possible_combinations = list(itertools.product(range(nb_classes), repeat=sequence_length))
    combination_probabilities = list()
    for combination in all_possible_combinations:
        combination_probability = 1
        for sequence_position, cls in enumerate(combination):
            combination_probability *= probabilities[:, sequence_position, cls]
        combination_probabilities.append(combination_probability)

    return all_possible_combinations, combination_probabilities


def calculate_p_value(sequence_probability: np.ndarray, sequence_probabilities: np.ndarray) -> float:
    x = stats.percentileofscore(sequence_probabilities, sequence_probability)
    return float(1) - (float(x) / float(100))


def plot_probability_matrix(probability_matrix: np.ndarray):
    for iTrial, trial in enumerate(probability_matrix):
        sns.heatmap(np.swapaxes(trial, axis1=0, axis2=1), cmap="YlGnBu", annot=False)
        plt.savefig(f"trial_{iTrial}.png")


if __name__ == '__main__':
    results_path = "/mnt/3b5a15cf-20ff-4840-8d84-ddbd428344e9/ALAB1/philippe/output_ml/train_CTF_avg/results.pkl"
    results = read_pkl.read(results_path)

    #time_gen_matrix = results['logistic_regression']['time_generalization']['replication_1']['temp_shiva'].time_generalization_matrix
    #max_time_gen_matrix = np.max(time_gen_matrix, axis=1)
    #iBest = np.argmax(max_time_gen_matrix)
    #pip = results['logistic_regression']['time_generalization']['replication_1']['temp_shiva'].time_generalization_pipelines[iBest]

    pip = results["logistic_regression"]["time_generalization"]["replication_1"]["temp_shiva"].trained_pipeline

    d = "/mnt/7012ad01-e4b3-459b-9944-8395b35698e6/ALAB/temp_shiva/seq_CTF/All_WM1/data_trial_1.mat"
    X1 = read_mat.read_mat(d, mat_file_key="F")
    X1 = X1[:, 501:3001]

    d = "/mnt/7012ad01-e4b3-459b-9944-8395b35698e6/ALAB/temp_shiva/seq_CTF/All_WM1/data_trial_1.mat"
    X2 = read_mat.read_mat(d, mat_file_key="F")
    X2 = X2[:, 3001:5501]

    X = np.stack([X1, X2])

    y1 = np.array([2, 4, 1])
    y2 = np.array([1, 2, 4])
    y = np.stack([y1, y2])

    output = time_sequenceness(pip, X, y)
    save_pkl.save(output, "../models/linear_models/output.pkl")
    save_mat.save(output, "../models/linear_models/output.mat")

    print("Done")
