import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os


class SequencenessOutput:

    def __init__(self, possible_sequences: list, compression_factor: int, expected_sequences: np.ndarray,
                 frames: tuple, data_trial_filepaths: List[str]):
        self.possible_sequences = possible_sequences
        self.compression_factor = compression_factor
        self.expected_sequences = expected_sequences
        self.frames = frames
        self.data_trial_filepaths = data_trial_filepaths

        self.python_index_expected_sequence = self.__get_index_of_sequences__(self.expected_sequences)
        self.python_index_reversed_sequence = self.__get_index_of_sequences__(self.expected_sequences[::-1])
        self.matlab_index_expected_sequence = self.python_index_expected_sequence + 1
        self.matlab_index_reversed_sequence = self.python_index_reversed_sequence + 1
        self.frames_of_compression_factor = []
        self.predicted_sequences: np.ndarray = np.array([])
        self.probabilities: np.ndarray = np.array([])
        self.sequences_count = np.array([[0 for _ in range(len(self.possible_sequences))]] * len(self.data_trial_filepaths))

    def add_sequence_prediction(self, frame_of_compression_factor: int, predicted_sequences: np.ndarray,
                                probabilities: np.ndarray):
        self.frames_of_compression_factor.append(frame_of_compression_factor)

        for iTrial, predicted_sequence in enumerate(predicted_sequences):

            index_of_predicted_sequences = self.__get_index_of_sequences__(predicted_sequence)
            self.sequences_count[iTrial][index_of_predicted_sequences] += 1

        predicted_sequences = np.expand_dims(predicted_sequences, axis=0)
        probabilities = np.expand_dims(probabilities, axis=0)
        if len(self.predicted_sequences) == 0:
            self.predicted_sequences = predicted_sequences
            self.probabilities = probabilities
        else:
            self.predicted_sequences = np.vstack((self.predicted_sequences, predicted_sequences))
            self.probabilities = np.vstack((self.probabilities, probabilities))

    def plot_sequence_count(self, output_folder: str):

        for iTrial in range(len(self.data_trial_filepaths)):

            trial_filepath = self.data_trial_filepaths[iTrial]
            trial_filename = os.path.splitext(os.path.basename(trial_filepath))[0]

            title = f"Sequence Count - {trial_filename} - Compression factor = {self.compression_factor}"

            fig, ax = plt.subplots(figsize=(15, 10))

            plt.plot(self.sequences_count[iTrial])
            plt.axvline(x=self.python_index_expected_sequence[iTrial],
                        color='b', linestyle='dashed', label='expected sequence')
            plt.axvline(x=self.python_index_reversed_sequence[iTrial],
                        color='r', linestyle='dashed', label='reversed sequence')
            plt.title(title)
            plt.legend()

            plt.tight_layout()

            figure_filename = f"{trial_filename}_cf_{self.compression_factor}.png"
            figure_path = os.path.join(output_folder, figure_filename)
            plt.savefig(figure_path)
            plt.close()

    def __get_index_of_sequences__(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.ndim == 1:
            sequences = np.expand_dims(sequences, axis=0)
        indices = np.array([np.where((self.possible_sequences == element).all(axis=1))[0] for element in sequences])
        return indices.squeeze()

