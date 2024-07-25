import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
from tqdm import tqdm

from util import save_mat, util


class TrainingOutput:

    def __init__(self):

        self.subject = None
        self.random_state = None

        # Pipeline
        self.trained_pipeline = None
        self.p_value = None

        # Data
        self.channel_indexes = None
        self.data_trials_filepaths = None

        # Cross validation
        self._cross_validation_output = None
        self._number_of_folds = None
        self._standard_deviation_of_validation_accuracy = None

        # Accuracies
        self.training_accuracy = None
        self.validation_accuracy = None
        self.test_accuracy = None

        # Time Generalization
        self.time_generalization_matrix = None
        self.time_generalization_pipelines = None

        self.grid_search_output = None

        # Feature Importance
        self.feature_importance = None

        # Permutation
        self.permutation_scores = None
        self.p_value = None

        # Filename when creating figures
        self._file_name = None
        self._figure_title = None

        # Training tag
        self.training_tag = None
        self.training_title = None

    def set_p_value(self, X, y):
        if self.training_tag == 'time_generalization':
            #trained_pipelines = self.time_generalization_pipelines
            trained_pipelines = [self.trained_pipeline]
        else:
            trained_pipelines = [self.trained_pipeline]

        self.p_value = []
        print("Calculating p-values")
        for pipeline in tqdm(trained_pipelines):
            _, _, p_value = pipeline.calculate_p_value_from_permutations(X, y)
            self.p_value.append(p_value)

    @property
    def file_name(self):
        if self._file_name is None:
            pipeline_name = self.trained_pipeline.name.title() if self.trained_pipeline is not None else None
            self._file_name = util.create_filename(
                (pipeline_name, self.training_tag), '_')
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        self._file_name = value

    @property
    def figure_title(self):
        if self._figure_title is None:
            pipeline_name = self.trained_pipeline.name.title() if self.trained_pipeline is not None else None
            self._figure_title = util.create_filename(
                (pipeline_name, self.training_title), ' - ')
        return self._figure_title

    @figure_title.setter
    def figure_title(self, value):
        self._figure_title = value

    @property
    def cross_validation_output(self):
        return self._cross_validation_output

    @cross_validation_output.setter
    def cross_validation_output(self, value):
        self._cross_validation_output = value

        iBest = np.argmax(self._cross_validation_output['test_score'])
        self.trained_pipeline = self._cross_validation_output['estimator'][iBest]
        self.training_accuracy = self._cross_validation_output['train_score'][iBest]
        self.validation_accuracy = self._cross_validation_output['test_score'][iBest]
        self._number_of_folds = len(self._cross_validation_output['test_score'])

    @classmethod
    def filter_list_of_training_outputs(cls, training_outputs, **filters):
        training_outputs = cls.__convert_to_list(training_outputs)

        filtered_training_outputs = list()
        for training_output in training_outputs:
            flag = True
            for key, value in filters.items():
                if getattr(training_output, key) != value:
                    flag = False
                    break
            if flag:
                filtered_training_outputs.append(training_output)

        return filtered_training_outputs

    def create_mat_file_for_nifti(self, filepath: str):
        if not filepath.endswith('.mat'):
            filepath += '.mat'

        folder = os.path.dirname(filepath)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        keys_to_keep = ['subject', 'feature_importance', 'channel_indexes']
        training_output_dico = {key: getattr(self, key) for key in keys_to_keep}

        save_mat.save(training_output_dico, filepath)

    @classmethod
    def export_to_mat_file(cls, trainings: dict, filepath: str):
        def replace_none_with_nan(d):
            """
            Recursively replace None in dictionary with np.nan.
            """
            for k, v in d.items():
                if v is None:
                    d[k] = float('nan')  # Or np.nan if numpy is available and imported
                elif isinstance(v, dict):
                    d[k] = replace_none_with_nan(v)
                elif isinstance(v, TrainingOutput):
                    d[k] = replace_none_with_nan(v.__dict__)
                elif k in ['trained_pipeline', 'estimator', 'time_generalization_pipelines']:
                    d[k] = 'Not converted to MAT file.'
            return d

        trainings = replace_none_with_nan(trainings)
        save_mat.save(trainings, filepath)

    @classmethod
    def plot_feature_importance_z_scored(cls, trainings):
        trainings = cls.__convert_to_list(trainings)

        figure = plt.figure()
        for training in trainings:
            plt.plot(training.feature_importance)

        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.show()
        return figure

    def plot_p_values(self):

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(self.p_value, marker="o")

        # Add labels
        plt.xlabel("trained pipeline")
        plt.ylabel("P-Value")

        plt.tight_layout()
        return fig

    @classmethod
    def plot_accuracies(cls, trainings):
        trainings = cls.__convert_to_list(trainings)

        bar_width = 0.25
        opacity = 0.4

        metrics = ['training_accuracy', 'validation_accuracy', 'test_accuracy']
        metric_colors = ['k', 'b', 'r']

        # Remove metric with None value (based on the first training)
        first_training = trainings[0]
        metrics = [metric for metric in metrics if getattr(first_training, metric) is not None]
        x_labels = []

        fig, ax = plt.subplots(figsize=(15, 10))
        for iTraining, training in enumerate(trainings):

            # Get x label for the current training
            subject = getattr(training, 'subject')
            if subject is None:
                x_labels.append(f'Training #{iTraining}')
            else:
                x_labels.append(subject)

            for iMetric, metric in enumerate(metrics):

                metric_value = getattr(training, metric)
                if metric_value is None:
                    continue

                x_position = iTraining + (iMetric * bar_width)
                bar = ax.bar(x=x_position, height=metric_value, width=bar_width, color=metric_colors[iMetric],
                             alpha=opacity)

                # Display bar value on bar
                ax.bar_label(bar, label_type='center', fmt='%.2f')

        # Add chance level line
        nb_classes = first_training.trained_pipeline.get_number_of_classes()
        plt.axhline(y=1/nb_classes, color='k', linestyle='dashed', label='chance level')
        metrics = ['chance level'] + metrics

        # Add legend
        plt.gca().legend(metrics, loc='lower right')

        # Add x labels
        plt.xticks(np.arange(len(trainings)) + bar_width / 2, x_labels, fontsize=10)
        plt.ylabel('%')

        plt.tight_layout()
        return fig

    def view_time_generalization(self):
        fig = plt.figure()
        ax = sn.heatmap(
            data=self.time_generalization_matrix, cmap="RdBu_r",
            vmin=np.min(self.time_generalization_matrix),
            vmax=np.max(self.time_generalization_matrix))
        ax.invert_yaxis()
        plt.xlabel('Trained models')
        plt.ylabel('Frame')
        plt.tight_layout()
        return fig

    def __attribute_to_string(self, attribute: str):
        attribute_value = getattr(self, attribute)

        s = f'{attribute.replace("_", " ").title().strip()}: '
        if isinstance(attribute_value, float):
            s += f'{attribute_value:0.2f}'
        else:
            s += f'{attribute_value}'
        return s + '\n'

    @classmethod
    def __convert_to_list(cls, training):
        if isinstance(training, TrainingOutput):
            output = [training]
        elif isinstance(training, list):
            output = training
        else:
            raise Exception('Unrecognized input!')
        return output

    def __str__(self):
        attributes_to_print = \
            ['subject', 'random_state', '_number_of_folds', 'training_accuracy', 'validation_accuracy',
             'test_accuracy']
        s = ''
        for attribute in attributes_to_print:
            s += self.__attribute_to_string(attribute)
        return s


if __name__ == "__main__":
    pass
