import copy
from tqdm import tqdm
import numpy as np
import sys

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from training.training_output import TrainingOutput


def time_generalization(X_train, y_train, X_test, y_test, pipeline, n_splits: int = 5, random_state: int = None):
    """
    For each time sample, a model is trained in cross-validation.
    The model is then validated on every time sample of the validation data.
    :param n_splits:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param pipeline:
    :param random_state:
    :return:
    """

    # Initialize variables
    trained_pipelines = list()
    validation_scores = list()

    # Loop over every time sample
    for iTime_sample in tqdm(range(X_train.shape[2]), file=sys.stdout):

        # Instantiate pipeline
        current_pipeline = copy.deepcopy(pipeline)

        # Select all Xs at one timestamp
        current_X_train = X_train[:, :, iTime_sample]

        # (nb_epochs, nb_channels) -> (nb_epochs, nb_channels, 1)
        current_X_train = np.expand_dims(current_X_train, axis=2)

        # Cross-validate
        s_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cross_validation_output = cross_validate(current_pipeline, current_X_train, y_train, cv=s_k_fold,
                                                 return_estimator=True, return_train_score=True)

        # Select fitted pipeline with the best validation accuracy ("test score")
        iBest = np.argmax(cross_validation_output['test_score'])
        best_pipeline = cross_validation_output['estimator'][iBest]
        trained_pipelines.append(best_pipeline)

        # Initialize variable
        ith_time_sample_validation_scores = list()

        # Loop over every time sample
        for jTime_sample in range(X_test.shape[2]):
            current_X_validation = X_test[:, :, jTime_sample]

            # (nb_epochs, nb_channels) -> (nb_epochs, nb_channels, 1)
            current_X_validation = np.expand_dims(current_X_validation, axis=2)

            # Get score and store it
            jth_time_sample_validation_score = best_pipeline.score(current_X_validation, y_test)
            ith_time_sample_validation_scores.append(jth_time_sample_validation_score)

        validation_scores.append(ith_time_sample_validation_scores)

    validation_scores = np.array(validation_scores)

    # Pipeline predictions for one time sample as columns
    validation_scores = np.swapaxes(validation_scores, axis1=0, axis2=1)

    # Select best pipeline
    pipeline_max_accuracy = np.max(validation_scores, axis=0)
    iBest = np.argmax(pipeline_max_accuracy)
    best_pipeline = trained_pipelines[iBest]

    output = TrainingOutput()
    output.time_generalization_matrix = validation_scores
    output.time_generalization_pipelines = trained_pipelines
    output.trained_pipeline = best_pipeline
    output.training_tag = "time_generalization"
    output.training_title = "Time Generalization"
    return output


def basic_cross_val(X_train, y_train, X_test, y_test, pipeline, n_splits: int = 5, random_state: int = None):

    # Cross-validate
    s_k_fold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    cross_validation_output = cross_validate(pipeline, X_train, y_train, cv=s_k_fold,
                                             return_estimator=True, return_train_score=True)

    # Test models
    test_accuracies = list()
    for trained_pipeline in cross_validation_output['estimator']:
        test_accuracy = trained_pipeline.score(X_test, y_test)
        test_accuracies.append(test_accuracy)

    # Select best pipeline
    iBest = np.argmax(test_accuracies)
    best_pipeline = cross_validation_output['estimator'][iBest]
    test_accuracy = test_accuracies[iBest]

    # Store values in a TrainingOutput object
    output = TrainingOutput()
    output.cross_validation_output = cross_validation_output
    output.test_accuracy = test_accuracy
    output.trained_pipeline = best_pipeline
    output.training_tag = "cross_validation"
    output.training_title = "Cross Validation"

    return output


def cross_condition_training(X_train, y_train, X_test, y_test, pipeline):

    pipeline.fit(X_train, y_train)
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    output = TrainingOutput()
    output.trained_pipeline = pipeline
    output.training_accuracy = train_score
    output.test_accuracy = test_score

    return output


def grid_search(X_train, y_train, X_test, y_test, pipeline, pipeline_parameters, verbose=True):
    gs_cv = GridSearchCV(
            estimator=pipeline, param_grid=pipeline_parameters, cv=5,
            scoring='accuracy', refit=True, verbose=0)
    gs_cv.fit(X_train, y_train)

    best_pipeline = gs_cv.best_estimator_
    best_parameters = gs_cv.best_params_
    grid_search_accuracy = gs_cv.best_score_

    train_accuracy = best_pipeline.score(X_train, y_train)
    test_accuracy = best_pipeline.score(X_test, y_test)

    if verbose:
        print(f"Grid search best score: {grid_search_accuracy}")
        print(f"Best parameters: {best_parameters}")

    output = TrainingOutput()
    output.grid_search_output = gs_cv
    output.trained_pipeline = best_pipeline
    output.training_accuracy = train_accuracy
    output.validation_accuracy = grid_search_accuracy
    output.test_accuracy = test_accuracy
    output.training_tag = "grid_search"
    output.training_title = "Grid Search"

    return output
