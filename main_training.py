import os
import argparse
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from util import save_pkl, util, process_data
from bst2py import load_x_y
from training.training_output import TrainingOutput
from training.training_input import TrainingInput
from training import trainings


def parse_command_line() -> str:
    parser = argparse.ArgumentParser(description="Test pipelines on data.")
    parser.add_argument("-tif", "--training_input_file", dest="training_input_filepath",
                        type=str, nargs="?",
                        help="Path to a training input file.")
    namespace = vars(parser.parse_args())
    return namespace["training_input_filepath"]


if __name__ == "__main__":
    DEFAULT_INPUT_FILE = "configuration_files/as_config.ini"

    training_input_file_path = parse_command_line()
    if training_input_file_path is None:
        training_input_file_path = DEFAULT_INPUT_FILE
    training_input = TrainingInput.read(training_input_file_path)

    # ------ MAIN ----- #

    training_input.validate()

    # Data
    protocol_name = training_input.protocol_name
    protocol_directory = training_input.protocol_directory
    subjects = training_input.subjects
    condition = training_input.condition
    condition_as_str = training_input.condition_to_string()
    class_folders = training_input.class_folders
    class_folders_test = training_input.class_folders_test
    start_sample = training_input.start_sample
    end_sample = training_input.end_sample
    pipeline_pre_processing = training_input.pre_processing

    output_folder = training_input.output_folder
    pipelines = training_input.create_pipelines()

    # Types of training
    training_types = {
        "cross_validation": training_input.run_cross_validation,
        "time_generalization": training_input.run_time_generalization,
        "time_sampling": training_input.run_time_sampling,
        "grid_search": training_input.run_grid_search
    }

    # Training parameters
    nb_replications = training_input.nb_replications
    test_size = training_input.test_size
    n_splits = training_input.n_splits
    random_state = training_input.random_state

    # Frequencies to filter
    sampling_frequency = training_input.sampling_frequency
    high_freq = training_input.high_freq
    low_freq = training_input.low_freq

    # Permutation
    run_permutation = training_input.run_permutation
    n_permutations = training_input.n_permutations
    permutation_n_splits = training_input.permutation_n_splits

    # ----- MAIN ----- #
    print(f"Protocol: {protocol_name}")
    print(f"Condition: {condition}")
    print()

    # Create output folder
    output_folder = os.path.join(output_folder, condition_as_str)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Create 'figs' folder
    fig_folder = os.path.join(output_folder, "figs")
    if not os.path.isdir(fig_folder):
        os.makedirs(fig_folder)

    # Store all outputs
    results = dict()

    # Loop over pipelines
    for pipeline in pipelines:
        print(f"Pipeline: {pipeline.name.title()}")
        results[pipeline.name] = dict()

        # Loop over trainings
        for training_key in training_types.keys():
            print(f"Training type: {training_key}")
            if not training_types[training_key]:
                print(f"Skipping training")
                print()
                continue
            results[pipeline.name][training_key] = dict()

            # Repeat training
            loaded_data = []
            for iRepeat in range(1, nb_replications + 1):
                results[pipeline.name][training_key][f"replication_{iRepeat}"] = dict()
                print(f"Training Starting [Iteration {iRepeat}/{nb_replications}] ({len(subjects)} subjects)")

                # Loop over subjects
                for iSubject, subject in enumerate(subjects):
                    print(f"Subject: {subject}")

                    # For the first training, read and process data
                    if iRepeat == 1:

                        # Get subject data directory
                        data_directory = os.path.join(protocol_directory, subject)

                        # Load train and test data
                        if len(condition) == 1:
                            data_directory = util.join_path(data_directory, condition[0])

                            data_trials = \
                                load_x_y.load_data_trials(data_directory, class_folder_names=class_folders)

                            labels = [data_trial.get_class_name() for data_trial in data_trials]

                            # Split data
                            data_trials_train, data_trials_test, _, _ = \
                                train_test_split(data_trials, labels,
                                                 test_size=test_size,
                                                 random_state=random_state,
                                                 stratify=labels, shuffle=True)

                        # If cross-condition
                        elif len(condition) == 2:
                            train_condition, test_condition = condition[0], condition[1]
                            train_class_folders, test_class_folders = class_folders, class_folders_test

                            # Get data directory
                            train_data_directory = os.path.join(protocol_directory, subject, train_condition)
                            test_data_directory = os.path.join(protocol_directory, subject, test_condition)

                            # Load data trials
                            data_trials_train = \
                                load_x_y.load_data_trials(train_data_directory, class_folder_names=train_class_folders)

                            data_trials_test = \
                                load_x_y.load_data_trials(test_data_directory, class_folder_names=test_class_folders)

                        else:
                            raise Exception(f"Invalid number of conditions. It is either one or two.")

                        X_train, y_train, class_labels_train = load_x_y.data_trials_to_x_y_labels(data_trials_train)
                        X_test, y_test, class_labels_test = load_x_y.data_trials_to_x_y_labels(data_trials_test)

                        data_trials_filepaths = \
                            [data_trial.get_filepath() for data_trial in data_trials_train + data_trials_test]
                        channel_indexes = \
                            [data_trial.get_channel_indexes() for data_trial in data_trials_train + data_trials_test]

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

                        # Store processed data for next training
                        loaded_data.append(
                            (X_train, X_test, y_train, y_test, class_labels, channel_indexes, data_trials_filepaths))

                    # Load data from 1st training
                    else:
                        X_train, X_test, y_train, y_test, class_labels, channel_indexes, data_trials_filepaths = \
                            loaded_data[iSubject]

                    # Graph distribution
                    if iRepeat == 1:
                        print("Plotting distribution")

                        process_data.graph_distribution(y_train, y_test, class_labels)
                        plt.title(util.create_filename(
                            (protocol_name.title(), condition_as_str, subject, "Classes Distribution"), " - "))
                        plt.tight_layout()

                        figure_filename = util.create_filename(
                            (protocol_name.title(), condition_as_str, subject, "distribution"), "_", ".png")
                        plt.savefig(os.path.join(fig_folder, figure_filename), orientation="landscape")
                        plt.close()

                    # ----- TRAINING ----- #

                    if training_key == "cross_validation":
                        print("Executing cross validation...")
                        output = trainings.basic_cross_val(
                            X_train, y_train, X_test, y_test, pipeline, n_splits=n_splits,
                            random_state=random_state)

                        if run_permutation:
                            print(f"Running {n_permutations} permutations ({n_splits} splits)")
                            trained_pipeline = output.trained_pipeline
                            _, permutation_scores, p_value = \
                                trained_pipeline.calculate_p_value_from_permutations(
                                    X=X_test, y=y_test, n_permutations=n_permutations, n_splits=n_splits
                                )
                            output.permutation_scores = permutation_scores
                            output.p_value = p_value

                    elif training_key == "time_generalization" and iRepeat == 1:
                        print("Executing time generalization...")
                        output = trainings.time_generalization(
                            X_train, y_train, X_test, y_test, pipeline, n_splits=n_splits,
                            random_state=random_state)

                    elif training_key == "time_sampling":
                        X_train_time_sample, y_train_time_sample = \
                            process_data.convert_epoch_to_timestamp_class_sample(X_train, y_train)
                        X_test_time_sample, y_test_time_sample = \
                            process_data.convert_epoch_to_timestamp_class_sample(X_test, y_test)

                        # Graph distribution of time sample data
                        process_data.graph_distribution(y_train_time_sample, y_test_time_sample, class_labels)
                        plt.title(
                            util.create_filename(
                                (protocol_name.title(), condition_as_str, subject, "Time Sample Classes Distribution"),
                                " - "))
                        plt.tight_layout()

                        # Save figure
                        figure_filename = \
                            util.create_filename(
                                (protocol_name.title(), condition_as_str, subject, "time_sample_distribution"),
                                "_", ".png")
                        plt.savefig(os.path.join(fig_folder, figure_filename), orientation="landscape")
                        plt.close()

                        output = trainings.basic_cross_val(
                            X_train_time_sample, y_train_time_sample, X_test_time_sample, y_test_time_sample,
                            pipeline, n_splits=n_splits, random_state=random_state)

                    elif training_key == "grid_search" and iRepeat == 1:
                        print("Executing grid search...")
                        output = trainings.grid_search(
                            X_train, y_train, X_test, y_test, pipeline,
                            pipeline.grid_search_parameters)

                    else:
                        continue

                    output.subject = subject
                    output.channel_indexes = channel_indexes
                    output.data_trials_filepaths = data_trials_filepaths
                    output.file_name = util.create_filename(
                        (protocol_name.title(), condition_as_str, subject, output.file_name), "_")
                    output.figure_title = util.create_filename(
                        (protocol_name.title(), condition_as_str, subject, output.figure_title), " - ")

                    results[pipeline.name][training_key]["replication_" + str(iRepeat)][subject] = output

                    # Save output as MAT to create Nifti
                    if training_key == "cross_validation":
                        trained_pipeline = output.trained_pipeline
                        output.feature_importance = \
                            trained_pipeline.extract_feature_importance()
                        mat_folder = os.path.join(output_folder, "mat4nii")
                        if not os.path.isdir(mat_folder):
                            os.makedirs(mat_folder)
                        mat_filename = util.create_filename((output.file_name, str(iRepeat)), "_")
                        output.create_mat_file_for_nifti(os.path.join(mat_folder, mat_filename + ".mat"))

                    # Create confusion matrices
                    if training_key == "cross_validation":
                        trained_pipeline = output.trained_pipeline
                        y_predicted = trained_pipeline.predict(X_test)
                        trained_pipeline.confusion_matrix(y_test, y_predicted)
                        plt.title(util.create_filename((output.figure_title, "Test"), " - "))
                        plt.tight_layout()

                        # Save figure
                        figure_filename = util.create_filename((output.file_name, "cm", str(iRepeat)), "_", ".png")
                        plt.savefig(os.path.join(fig_folder, figure_filename))
                        plt.close()

                # Create figures
                all_outputs = list(results[pipeline.name][training_key]["replication_" + str(iRepeat)].values())
                if not all_outputs:
                    continue

                if training_key == "cross_validation":
                    print("Plotting accuracies (basic training)...")

                    # Plot accuracies
                    TrainingOutput.plot_accuracies(all_outputs)
                    plt.title(util.create_filename(
                        (protocol_name.title(), condition_as_str, pipeline.name.title(), "Accuracies",
                         "Cross Validation"), " - "))
                    plt.tight_layout()

                    # Save figure
                    figure_filename = \
                        util.create_filename(
                            (protocol_name.title(), condition_as_str, pipeline.name.title(), training_key,
                             "accuracies", str(iRepeat)), "_", ".png")
                    plt.savefig(os.path.join(fig_folder, figure_filename))
                    plt.close()

                elif training_key == "time_generalization":
                    print("Plotting time generalization...")
                    for output in all_outputs:
                        output.view_time_generalization()
                        plt.title(output.figure_title)
                        plt.tight_layout()

                        plt.savefig(os.path.join(fig_folder, output.file_name + ".png"))
                        plt.close()

                elif training_key == "time_sampling":
                    TrainingOutput.plot_accuracies(all_outputs)
                    plt.title(
                        util.create_filename(
                            (protocol_name.title(), condition_as_str, pipeline.name.title(), "Accuracies",
                             "Time Sampling Training"), " - "))
                    plt.tight_layout()

                    # Save figure
                    figure_filename = \
                        util.create_filename(
                            (protocol_name.title(), condition_as_str, pipeline.name.title(), "time_sample",
                             str(iRepeat)),
                            "_", ".png")
                    plt.savefig(os.path.join(fig_folder, figure_filename))
                    plt.close()

                elif training_key == "grid_search":

                    # Plot accuracies
                    TrainingOutput.plot_accuracies(all_outputs)
                    plt.title(util.create_filename(
                        (protocol_name.title(), condition_as_str, pipeline.name.title(), "Accuracies",
                         "Grid Search"), " - "))
                    plt.tight_layout()

                    # Save figure
                    figure_filename = \
                        util.create_filename(
                            (protocol_name.title(), condition_as_str, pipeline.name.title(),
                             "grid_search", "accuracies", str(iRepeat)), "_", ".png")
                    plt.savefig(os.path.join(fig_folder, figure_filename))
                    plt.close()

                    grid_search_filename = "_".join(("grid_search_report_", protocol_name, pipeline.name)) + ".txt"
                    grid_search_filepath = os.path.join(output_folder, grid_search_filename)
                    f = open(grid_search_filepath, "w")
                    f.write(f"Protocol: {protocol_name}\n")
                    f.write(f"Pipeline: {pipeline.name}\n\n")
                    for output in all_outputs:
                        f.write(f"Subject: {output.subject}\n")
                        parameters = output.trained_pipeline.get_params()
                        parameters_str = "\n".join(f"\t{k}: {v}" for k, v in parameters.items())
                        f.write(f"Best parameters\n: {parameters_str}\n\n")

    save_pkl.save(results, os.path.join(output_folder, "results.pkl"))
    TrainingOutput.export_to_mat_file(results, os.path.join(output_folder, "results.mat"))
    print(f"Training finished:\n{output_folder}")
