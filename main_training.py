import os
import sys
import argparse
import shutil

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from util import save_pkl, util, process_data
from bst2py import load_x_y, bst_protocol
from training.training_output import TrainingOutput
from training import trainings
from configuration.training_configuration import TrainingConfiguration


def parse_command_line() -> str:
    parser = argparse.ArgumentParser(description="Test pipelines on data.")
    parser.add_argument("-tif", "--training_input_file", dest="training_input_filepath",
                        type=str, nargs="?",
                        help="Path to a training input file.")
    namespace = vars(parser.parse_args())
    return namespace["training_input_filepath"]


if __name__ == "__main__":
    DEFAULT_INPUT_FILE = "../configuration_files/arthur_training.ini"
    DEFAULT_INPUT_FILE = "/mnt/3b5a15cf-20ff-4840-8d84-ddbd428344e9/ALAB1/rg/Arthur/ml_corentin/arthur_training.mat"

    training_input_file_path = parse_command_line()
    if training_input_file_path is None:
        training_input_file_path = DEFAULT_INPUT_FILE
    training_configuration = TrainingConfiguration.read(training_input_file_path)

    # ------ MAIN ----- #

    training_configuration.validate()

    # Data
    protocol_name = training_configuration.protocol_name
    protocol_directory = training_configuration.protocol_directory
    subject_patterns_to_keep = training_configuration.subjects_to_keep
    subject_patterns_to_skip = training_configuration.subjects_to_skip
    train_studies = training_configuration.train_studies
    test_studies = training_configuration.test_studies
    start_sample = training_configuration.start_sample
    end_sample = training_configuration.end_sample
    pipeline_pre_processing = training_configuration.pre_processing

    output_folder = training_configuration.output_folder
    pipelines = training_configuration.create_pipelines()

    # Types of training
    training_types = training_configuration.get_training_types()

    # Training parameters
    nb_replications = training_configuration.nb_replications
    test_size = training_configuration.test_size
    n_splits = training_configuration.n_splits
    random_states = training_configuration.random_states

    # Frequencies to filter
    sampling_frequency = training_configuration.sampling_frequency
    high_freq = training_configuration.high_freq
    low_freq = training_configuration.low_freq

    # Permutation
    run_permutation = training_configuration.run_permutation
    n_permutations = training_configuration.n_permutations
    permutation_n_splits = training_configuration.permutation_n_splits

    # ----- MAIN ----- #
    protocol_name_titled = protocol_name.title()
    protocol_name_tag = protocol_name.replace(" ", "_").lower()

    if random_states is None:
        random_states = [None] * nb_replications
    else:
        nb_replications = len(random_states)

    # Create output folder
    if os.path.isdir(output_folder):
        raise Exception(f"Delete the output folder with the same name first:\n{output_folder}")
    else:
        os.makedirs(output_folder)

    # Create 'figs' folder
    fig_folder = os.path.join(output_folder, "figs")
    if not os.path.isdir(fig_folder):
        os.makedirs(fig_folder)

    mat4nii_folder = os.path.join(output_folder, "mat4nii")
    if not os.path.isdir(mat4nii_folder):
        os.makedirs(mat4nii_folder)

    print(f"Protocol: {protocol_name_titled}")

    # Copy configuration file
    config_folder = os.path.join(output_folder, "configuration_file")
    if not os.path.isdir(config_folder):
        os.makedirs(config_folder)
    training_input_filename = os.path.basename(training_input_file_path)
    new_training_input_filepath = os.path.join(config_folder, training_input_filename)
    shutil.copyfile(training_input_file_path, new_training_input_filepath)

    # Load data for each subject
    bst_protocol = bst_protocol.BstProtocol(
        directory=protocol_directory,
        subject_patterns_to_keep=subject_patterns_to_keep,
        subject_patterns_to_remove=subject_patterns_to_skip,
        train_studies=train_studies,
        test_studies=test_studies
    )
    subjects, train_data_trials, test_data_trials = bst_protocol.get_data_trials()

    outputs = []
    for iReplication in range(1, nb_replications + 1):
        random_state = random_states[iReplication-1]

        for iSubject in range(len(subjects)):
            subject = subjects[iSubject]

            if test_data_trials[iSubject] is None:
                labels = [train_data_trial.get_class_name() for train_data_trial in train_data_trials[iSubject]]

                # Split data
                train_data_trials[iSubject], test_data_trials[iSubject], _, _ = \
                    train_test_split(train_data_trials[iSubject], labels,
                                     test_size=test_size,
                                     random_state=random_state,
                                     stratify=labels, shuffle=True)

            X_train, y_train, class_labels_train = load_x_y.data_trials_to_x_y_labels(train_data_trials[iSubject])
            X_test, y_test, class_labels_test = load_x_y.data_trials_to_x_y_labels(test_data_trials[iSubject])

            data_trials_filepaths = \
                [data_trial.get_filepath() for data_trial in train_data_trials[iSubject] + test_data_trials[iSubject]]

            channel_indexes = \
                [data_trial.get_channel_indexes() for data_trial in
                 train_data_trials[iSubject] + test_data_trials[iSubject]]

            if class_labels_train != class_labels_test:
                raise Exception(f"'class_labels_train' and 'class_labels_test' should be equal.")
            class_labels = class_labels_test

            # Crop data in time
            if start_sample is not None or end_sample is not None:
                X_train = \
                    process_data.slice_array(X_train, axis=2, index_1=start_sample, index_2=end_sample)
                X_test = \
                    process_data.slice_array(X_test, axis=2, index_1=start_sample, index_2=end_sample)

            # Filter data
            if low_freq is not None or high_freq is not None:
                print("Filtering...")

                X_train = process_data.band_pass_filter(
                    X_train, sampling_frequency=sampling_frequency, low_freq=low_freq,
                    high_freq=high_freq,
                    verbose=False)
                X_test = process_data.band_pass_filter(
                    X_test, sampling_frequency=sampling_frequency, low_freq=low_freq,
                    high_freq=high_freq,
                    verbose=False)

            # Graph distribution
            figure_filename = util.create_filename((protocol_name, subject, "distribution"), "_", ".png")
            if not os.path.isfile(figure_filename):
                process_data.graph_distribution(y_train, y_test)
                plt.savefig(os.path.join(fig_folder, figure_filename), orientation="landscape")
                plt.close()

            # Create class label-index mapping
            class_mapping_filepath = os.path.join(output_folder, "class_labels_indices_mapping.csv")
            if not os.path.isfile(class_mapping_filepath):
                process_data.create_class_label_index_mapping_file(class_labels, class_mapping_filepath)

            # Loop over trainings
            for training_type in training_types:

                # Loop over pipelines
                for pipeline in pipelines:

                    print(f"\nPIPELINE: {pipeline.name.replace('_', ' ').title()}")
                    print(f"TRAINING: {training_type.tag().replace('_', ' ').title()}")
                    print(f"REPLICATION: {iReplication}/{nb_replications}")
                    print(f"NB SUBJECTS: {len(subjects)}")

                    if training_type == trainings.TrainingType.CROSS_VALIDATION:
                        output = trainings.basic_cross_val(
                            X_train, y_train, X_test, y_test, pipeline, n_splits=n_splits,
                            random_state=random_state)

                    elif training_type == trainings.TrainingType.TIME_GENERALIZATION and iReplication == 1:
                        output = trainings.time_generalization(
                            X_train, y_train, X_test, y_test, pipeline, n_splits=n_splits,
                            random_state=random_state)
                        output.plot(output_folder)

                    else:
                        raise Exception("Invalid training type.")

                    output.subject = subject
                    output.i_replication = iReplication
                    output.data_trials_filepaths = data_trials_filepaths
                    output.channel_indexes = channel_indexes
                    output.random_state = random_state

                    output.create_mat_file_for_nifti(mat4nii_folder)

                    if run_permutation:
                        print(f"\nRunning {n_permutations} permutations ({permutation_n_splits} splits)")
                        trained_pipeline = output.trained_pipeline
                        _, permutation_scores, p_value = \
                            trained_pipeline.calculate_p_value_from_permutations(
                                X=X_test, y=y_test, n_permutations=n_permutations, n_splits=n_splits,
                                random_state=random_state
                            )
                        output.permutation_scores = permutation_scores
                        output.p_value = p_value

                    # Plot confusion matrix
                    trained_pipeline = output.trained_pipeline
                    filepath = os.path.join(fig_folder, f"{output.get_filename()}_cm.png")
                    trained_pipeline.plot_confusion_matrix(X_test, y_test, filepath)

                    outputs.append(output)

    # Plot cross-validation accuracies across subjects
    pipeline_names = [pipeline.name for pipeline in pipelines]
    for iReplication in range(1, nb_replications + 1):
        for pipeline_name in pipeline_names:
            cross_validation_outputs = TrainingOutput.filter_list_of_training_outputs(
                outputs,
                training_type=trainings.TrainingType.CROSS_VALIDATION,
                i_replication=iReplication,
                trained_pipeline_name=pipeline_name)

            filepath = os.path.join(fig_folder, f"{pipeline_name}_cross_validation_accuracies_rep_{iReplication}")
            trainings.CrossValidationOutput.plot_across_subjects(cross_validation_outputs, filepath)

        # elif training_key == "grid_search":
        #
        #     # Plot accuracies
        #     TrainingOutput.plot_accuracies(all_outputs)
        #     plt.title(util.create_filename(
        #         (protocol_name_titled, pipeline.name.title(), "Accuracies",
        #          "Grid Search"), " - "))
        #     plt.tight_layout()
        #
        #     # Save figure
        #     figure_filename = \
        #         util.create_filename(
        #             (protocol_name, pipeline.name,
        #              "grid_search", "accuracies", str(iRepeat)), "_", ".png")
        #     plt.savefig(os.path.join(fig_folder, figure_filename))
        #     plt.close()
        #
        #     grid_search_filename = "_".join(("grid_search_report_", protocol_name, pipeline.name)) + ".txt"
        #     grid_search_filepath = os.path.join(output_folder, grid_search_filename)
        #     f = open(grid_search_filepath, "w")
        #     f.write(f"Protocol: {protocol_name}\n")
        #     f.write(f"Pipeline: {pipeline.name}\n\n")
        #     for output in all_outputs:
        #         f.write(f"Subject: {output.subject}\n")
        #         parameters = output.trained_pipeline.get_params()
        #         parameters_str = "\n".join(f"\t{k}: {v}" for k, v in parameters.items())
        #         f.write(f"Best parameters\n: {parameters_str}\n\n")

    outputs_as_dico = TrainingOutput.list_to_dico(outputs)
    save_pkl.save(outputs_as_dico, os.path.join(output_folder, "results.pkl"))
    TrainingOutput.export_to_mat_file(outputs_as_dico, os.path.join(output_folder, "results.mat"))
    print(f"\nOUTPUT SAVED:\n{output_folder}")
