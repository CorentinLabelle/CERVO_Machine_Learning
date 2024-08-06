import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
from typing import List

from util import save_mat
from training.training_types import TrainingType


class TrainingOutput:

    def __init__(self):
        self.training_type = None

        self.subject = None
        self.i_replication = None
        self.data_trials_filepaths = None
        self.channel_indexes = None

        self.random_state = None

        # Pipeline
        self.trained_pipeline = None
        self.trained_pipeline_name = None
        self.feature_importance = None
        self.training_accuracy = None
        self.validation_accuracy = None
        self.test_accuracy = None

        # Permutation
        self.permutation_scores = None
        self.p_value = None

    def get_filename(self) -> str:
        subject = self.subject
        if self.subject != "":
            subject += "_"
        return f"{subject}{self.trained_pipeline_name}_{self.training_type.tag()}_rep_{self.i_replication}"

    def create_mat_file_for_nifti(self, folder: str):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        filepath = os.path.join(folder, f"{self.get_filename()}.mat")

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
                elif isinstance(v, TrainingType):
                    d[k] = v.tag()
            return d

        trainings = replace_none_with_nan(trainings.copy())
        save_mat.save(trainings, filepath)

    @classmethod
    def filter_list_of_training_outputs(cls, training_outputs, **filters):
        training_outputs = cls.__convert_to_list__(training_outputs)

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

    @classmethod
    def __convert_to_list__(cls, training):
        if isinstance(training, TrainingOutput):
            output = [training]
        elif isinstance(training, list):
            output = training
        else:
            raise Exception('Unrecognized input!')
        return output

    @classmethod
    def list_to_dico(cls, trainings: List["TrainingOutput"]) -> dict:
        trainings_as_dico = dict()
        for training in trainings:
            dico = trainings_as_dico
            keys = [training.trained_pipeline.name, training.training_type.tag(),
                    training.subject, f"replication_{training.i_replication}"]
            for key in keys[:-1]:
                if key == "":
                    continue
                if key not in dico:
                    dico[key] = dict()
                dico = dico[key]
            dico[keys[-1]] = training
        return trainings_as_dico


class CrossValidationOutput(TrainingOutput):

    def __init__(self, cross_validation_output, test_accuracies: np.ndarray):
        super().__init__()
        self.training_type = TrainingType.CROSS_VALIDATION
        self.cross_validation_output = cross_validation_output
        self.test_accuracies: np.ndarray = test_accuracies

        self.__select_best_pipeline__()

    def __select_best_pipeline__(self):
        iBest = np.argmax(self.test_accuracies)
        self.trained_pipeline = self.cross_validation_output['estimator'][iBest]
        self.trained_pipeline_name = self.trained_pipeline.name
        self.training_accuracy = self.cross_validation_output['train_score'][iBest]
        self.validation_accuracy = self.cross_validation_output['test_score'][iBest]
        self.test_accuracy = self.test_accuracies[iBest]
        self.feature_importance = self.trained_pipeline.extract_feature_importance()

    @classmethod
    def plot_across_subjects(cls, outputs: List["CrossValidationOutput"], filepath: str):
        bar_width = 0.25
        opacity = 0.4

        metrics = ['training_accuracy', 'validation_accuracy', 'test_accuracy']
        metric_colors = ['k', 'b', 'r']

        # Remove metric with None value (based on the first training)
        first_training = outputs[0]
        metrics = [metric for metric in metrics if getattr(first_training, metric) is not None]
        x_labels = []

        fig, ax = plt.subplots(figsize=(15, 10))
        for iTraining, training in enumerate(outputs):

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
        plt.axhline(y=1 / nb_classes, color='k', linestyle='dashed', label='chance level')
        metrics = ['chance level'] + metrics

        # Add legend
        plt.gca().legend(metrics, loc='lower right')

        # Add x labels
        plt.xticks(np.arange(len(outputs)) + bar_width / 2, x_labels, fontsize=10)
        plt.ylabel('%')

        plt.tight_layout()

        plt.title("Cross-Validation Accuracies")
        plt.savefig(filepath)

        plt.close()


class TimeGeneralizationOutput(TrainingOutput):

    def __init__(self, trained_pipelines: list, validation_scores: np.ndarray):
        super().__init__()
        self.training_type = TrainingType.TIME_GENERALIZATION
        self.trained_pipelines: list = trained_pipelines
        self.validation_scores: np.ndarray = validation_scores

        self.__select_best_pipeline__()

    def __select_best_pipeline__(self):
        iBest = np.argmax(self.validation_scores, axis=0)
        self.trained_pipeline = self.trained_pipelines[iBest]
        self.trained_pipeline_name = self.trained_pipeline.name
        self.training_accuracy = 0
        self.validation_accuracy = 0
        self.feature_importance = self.trained_pipeline.extract_feature_importance()

    def plot(self, folder: str):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        filepath = os.path.join(folder, f"{self.get_filename()}.png")

        plt.figure()
        ax = sn.heatmap(
            data=self.validation_scores, cmap="RdBu_r",
            vmin=np.min(self.validation_scores),
            vmax=np.max(self.validation_scores))
        ax.invert_yaxis()
        plt.xlabel('Trained models')
        plt.ylabel('Frame')
        plt.title("Some title")
        plt.tight_layout()
        plt.savefig(filepath)


class GridSearchOutput(TrainingOutput):

    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    pass
