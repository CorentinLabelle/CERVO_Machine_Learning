import argparse
import os

from bst2py import load_x_y
from sequenceness.sequenceness import *
from sequenceness.sequenceness_input import SequencenessInput
from training.training_output import TrainingOutput


def parse_command_line() -> str:
    parser = argparse.ArgumentParser(description="Test sequenceness.")
    parser.add_argument("-sif", "--sequenceness_input_file", dest="sequenceness_input_filepath",
                        type=str, nargs="?",
                        help="Path to a sequenceness input file.")
    namespace = vars(parser.parse_args())
    return namespace["sequenceness_input_filepath"]


if __name__ == "__main__":
    DEFAULT_INPUT_FILE = \
        "configuration_files/arthur_sequenceness.ini"

    sequenceness_input_file_path = parse_command_line()
    if sequenceness_input_file_path is None:
        sequenceness_input_file_path = DEFAULT_INPUT_FILE
    sequenceness_input = SequencenessInput.read(sequenceness_input_file_path)

    # MAIN #

    output_folder = sequenceness_input.output_folder
    data_folder = sequenceness_input.data_folder
    label_folder = sequenceness_input.label_folder
    label_starting_at_1 = sequenceness_input.label_index_starting_at_1
    results_path = sequenceness_input.result_filepath
    fields_to_trained_pipeline = sequenceness_input.pipeline_fields_in_result_file
    start_sample = sequenceness_input.start_sample
    end_sample = sequenceness_input.end_sample
    compression_factors = sequenceness_input.compression_factors

    # Create output folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # Load data trials
    data_trials = load_x_y.load_data_trials(data_folder)
    data_trials = data_trials[:5]

    # Load labels
    y = read_mat.read_mat(label_folder, mat_file_key="label_1")
    y = y.astype(int)
    y = y[:5]
    if label_starting_at_1:
        y = y - 1

    # Load pipeline
    results = read_pkl.read(results_path)

    dico = results
    for key in fields_to_trained_pipeline:
        dico = dico[key]
    training_output: TrainingOutput = dico
    trained_pipeline = training_output.trained_pipeline

    outputs = time_sequenceness(
        trained_pipeline, data_trials, y,
        compression_factors=compression_factors, frame_1=start_sample, frame_2=end_sample)

    # Plot output
    for output in outputs:
        output.plot_sequence_count(output_folder)

    save_mat.save(outputs, os.path.join(output_folder, "sequenceness_output.mat"))
    save_pkl.save(outputs, os.path.join(output_folder, "sequenceness_output.pkl"))

    print("Done")
