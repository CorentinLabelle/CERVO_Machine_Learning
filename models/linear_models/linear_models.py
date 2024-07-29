import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn import pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import permutation_test_score
from sklearn.svm import SVC

from scipy import stats

import mne.decoding
from mne.decoding import LinearModel, get_coef, Vectorizer

from util import sklearn_custom_steps
from util import save_pkl


def __get_pre_processing_steps__(pre_processing: str) -> list:
    pre_processing = pre_processing.upper()

    if pre_processing == "VECTORIZE":
        step_1 = ('vectorize', Vectorizer())
    elif pre_processing == "AVERAGE":
        step_1 = ('time_average', sklearn_custom_steps.TimeAverage())
    else:
        raise Exception(f"Invalid pre_processing: {pre_processing}.")

    step_2 = ('scaling', StandardScaler())

    return [step_1, step_2]


class PipelineFactory:

    @classmethod
    def create(cls, pipeline_type: str, pre_processing: str) -> "BaseLinearPipeline":
        pipeline_type = pipeline_type.upper()
        if pipeline_type == "RIDGE":
            return CustomRidge(pre_processing=pre_processing)
        elif pipeline_type == "LOGISTIC":
            return CustomLogisticRegression(pre_processing=pre_processing)
        elif pipeline_type == "PERCEPTRON":
            return CustomPerceptron(pre_processing=pre_processing)
        elif pipeline_type == "SVC":
            return CustomSVC(pre_processing=pre_processing)


class BaseLinearPipeline(pipeline.Pipeline):
    """
    This class inherits from the sklearn.pipeline.Pipeline class.
    """

    def __init__(self, pre_processing: str, classifier_step: tuple,
                 name: str, parameters: dict = None):
        self.pre_processing = pre_processing
        self.classifier_step = classifier_step
        self.parameters = parameters
        self.name = name

        steps = __get_pre_processing_steps__(pre_processing)
        steps.append(classifier_step)
        super().__init__(steps)

        if parameters is not None:
            self.set_params(**parameters)

    # Override set_params method from parent class (sklearn.pipeline.Pipeline)
    def set_params(self, **pipeline_parameters: dict):
        super().set_params(**pipeline_parameters)
        return self

    def confusion_matrix(self, true_labels, predictions):
        cm = confusion_matrix(true_labels, predictions)
        cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        class_labels = self.get_class_labels()
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_labels)
        cm_display.plot()
        return cm

    def plot_confusion_matrix(self, X, y, filepath: str):
        y_predicted = self.predict(X)
        self.confusion_matrix(y, y_predicted)
        plt.title("Confusion Matrix")
        plt.tight_layout()

        # Save figure
        plt.savefig(filepath)
        plt.close()

    def classification_report(self, true_labels, predictions):
        target_names = [str(class_label) for class_label in self.classes_]
        report = classification_report(true_labels, predictions, target_names=target_names)
        print(report)
        return report

    def extract_feature_importance(self, coefficient_attribute: str = 'patterns'):
        """
        SOURCE: https://mne.tools/stable/generated/mne.decoding.get_coef.html

        coefficient_attribute:
            'patterns':spatial patterns
            'filters': spatial filters
        """

        valid_coefficient_attributes = ['patterns', 'filters']

        coefficient_attribute = coefficient_attribute.lower()
        if coefficient_attribute not in valid_coefficient_attributes:
            valid_attributes_str = ', '.join(valid_coefficient_attributes)
            raise Exception(f'Invalid coefficient_attribute. Valid attributes: {valid_attributes_str}')

        coefficient_attribute = coefficient_attribute + '_'
        coefficients = get_coef(self, attr=coefficient_attribute, inverse_transform=True)
        return np.absolute(coefficients)

    def extract_z_scored_feature_importance(self):
        # Feature importance: (nb_classes, nb_channels)
        feature_importance = self.extract_feature_importance()
        feature_importance_averaged = np.mean(feature_importance, axis=0)
        return stats.zscore(feature_importance_averaged)

    def calculate_p_value_from_permutations(self, X, y,
                                            n_splits: int = 5, n_permutations: int = 50,
                                            random_state: int = None, scoring: str = 'accuracy'):

        [score, perm_score, p_value] = \
            permutation_test_score(self, X, y,
                                   random_state=random_state, scoring=scoring,
                                   cv=n_splits, n_permutations=n_permutations)

        return score, perm_score, p_value

    def predict(self, X, **predict_params):
        return super().predict(X, **predict_params)

    def predict_probabilities(self, X):
        """
        Input
            x: (nb_epochs, nb_channels, nb_timestamps)
        Return
            probability_matrix: (nb_epochs, nb_classes)
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        return self.predict_proba(X)

    def predict_sequence(self, X):
        """
        Input
            x: (nb_epochs, nb_channels, sequence_length)
        Return
            predictions: (nb_epochs, sequence_length)
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        # Loop directly on epoch and timestamp
        predictions = list()
        for iEpoch in range(len(X)):
            current_x = X[iEpoch]

            # Rearrange X to be (nb_timestamps, nb_channels, 1)
            current_x = np.swapaxes(current_x, axis1=0, axis2=1)
            current_x = np.expand_dims(current_x, axis=2)

            prediction = self.predict(current_x)
            predictions.append(prediction)

        return np.array(predictions)

    def predict_sequence_probabilities(self, X):
        """
        Input
            x: (nb_epochs, nb_channels, nb_timestamps)
        Return
            probability_matrix: (nb_epochs, nb_timestamps, nb_classes)
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        nb_timestamps = X.shape[2]
        probability_matrix = list()
        for iTimestamp in range(nb_timestamps):
            current_x = X[:, :, iTimestamp]

            # Change shape from (nb_epochs, nb_channels) to (nb_epochs, nb_channels, 1)
            current_x = np.expand_dims(current_x, axis=2)

            tmp = self.predict_proba(current_x)
            probability_matrix.append(tmp)
        probabilities = np.array(probability_matrix)

        # Change shape from () are (nb_epochs, nb_timestamps, nb_classes)
        probability_matrix = np.swapaxes(probabilities, axis1=0, axis2=1)
        return probability_matrix

    def get_class_labels(self):
        """
        Get the class labels from the pipeline. The pipeline must have been trained.
        :return:
        """
        model = self.steps[-1][1]
        if isinstance(model, mne.decoding.LinearModel):
            model = model.model
        return model.classes_

    def get_number_of_classes(self):
        """
        Get the number of classes from the pipeline. The pipeline must have been trained.
        :return:
        """
        return len(self.get_class_labels())

    def get_possible_sequences(self, sequence_length: int):
        return list(itertools.product(range(self.get_number_of_classes()), repeat=sequence_length))

    def save(self, filepath: str):
        save_pkl.save(self, filepath)


class CustomRidge(BaseLinearPipeline):

    def __init__(self, pre_processing: str, parameters: dict = None, name: str = 'ridge'):
        classifier_step = ('ridge', LinearModel(RidgeClassifierCV()))
        super().__init__(pre_processing, classifier_step=classifier_step, name=name, parameters=parameters)

    def predict_proba(self, X: np.ndarray):
        # https://stackoverflow.com/questions/22538080/scikit-learn-ridge-classifier-extracting-class-probabilities
        if X.ndim != 3:
            raise Exception(f"X dimension should be 2, not {X.ndim}")

        probabilities = []
        for current_X in X:
            d = self.decision_function(current_X)[0]
            probs = np.exp(d) / np.sum(np.exp(d))
            probabilities.append(probs)
        return np.array(probabilities)


class CustomLogisticRegression(BaseLinearPipeline):

    def __init__(self, pre_processing: str, parameters: dict = None, name: str = 'logistic_regression'):
        classifier_step = ('logistic_regression', LinearModel(LogisticRegression(max_iter=200)))
        super().__init__(pre_processing, classifier_step=classifier_step, name=name, parameters=parameters)

        # grid_search_parameters = {
        #     'logistic_regression__model__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        #     'logistic_regression__model__penalty': ['l1', 'l2', 'elasticnet'],  # , None],
        #     'logistic_regression__model__C': [1, 10],
        #     'logistic_regression__model__multi_class': ['auto', 'ovr', 'multinomial'],
        #     'logistic_regression__model__dual': [True, False],
        #     'logistic_regression__model__l1_ratio': [None, 0.5],
        #     'logistic_regression__model__max_iter': [100]
        # }
        super().__init__(pre_processing, classifier_step=classifier_step, name=name, parameters=parameters)


class CustomSVC(BaseLinearPipeline):

    def __init__(self, pre_processing: str, parameters: dict = None, name: str = 'svc'):
        classifier_step = ('svc', SVC(kernel='linear', C=1, probability=True))
        super().__init__(pre_processing, classifier_step=classifier_step, name=name, parameters=parameters)

    def extract_feature_importance(self, coefficient_attribute: str = None):
        return self.steps[2][1].coef_


class CustomPerceptron(BaseLinearPipeline):

    def __init__(self, pre_processing: str, parameters: dict = None, name: str = 'perceptron'):
        classifier_step = ('perceptron', LinearModel(Perceptron(max_iter=2000)))
        super().__init__(pre_processing, classifier_step=classifier_step, name=name, parameters=parameters)


def __loca_audi_logistic_regression():
    model_parameters = dict()
    model_parameters['logistic_regression__model__solver'] = 'newton-cg'
    model_parameters['logistic_regression__model__C'] = 1
    model_parameters['logistic_regression__model__penalty'] = 'l2'
    model_parameters['logistic_regression__model__multi_class'] = 'auto'
    return CustomLogisticRegression(parameters=model_parameters, name='log_reg_loca_audi', pre_processing="Average")


if __name__ == '__main__':
    pass
