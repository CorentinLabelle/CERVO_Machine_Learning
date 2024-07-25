"""
Grid Search: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
Scoring Parameter: https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
"""

import inspect
import sys
import os

from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import \
    LogisticRegression, Perceptron, RidgeClassifierCV, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from mne.decoding import Vectorizer

from util import sklearn_custom_steps, process_data
from bst2py import load_x_y

RANDOM_STATE = None


def __logistic_regression():
    pipeline = Pipeline([
        ('time_average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('log_reg', LogisticRegression(max_iter=2000))
    ]
    )

    pipe_parameters = {
        'log_reg__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        'log_reg__penalty': ['l1', 'l2', 'elasticnet'], # , None],
        'log_reg__C': [1, 10],
        'log_reg__multi_class': ['auto', 'ovr', 'multinomial'],
        'log_reg__dual': [True],
        'log_reg__l1_ratio': [None, 0.5],
        'log_reg__max_iter': [1000]
    }

    #pipe_parameters = {}

    return pipeline, pipe_parameters


def __passive_aggressive():
    pipeline = Pipeline([
        ('time_average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', PassiveAggressiveClassifier())
    ]
    )

    pipe_parameters = {}

    return pipeline, pipe_parameters


def __sgd_classifier():
    pipeline = Pipeline([
        ('time_average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', SGDClassifier(loss='hinge'))
    ]
    )

    pipe_parameters = {
        'estimator__loss': ['log', 'hinge'],
    }

    return pipeline, pipe_parameters


def __ridge_cv():
    pipeline = Pipeline([
        ('time_average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', RidgeClassifierCV())]
    )

    pipe_parameters = {
        'estimator__alphas': [0, 0.1, 0.5, 1, 5, 10],
    }

    return pipeline, pipe_parameters


def __multilayer_perceptron():
    pipeline = Pipeline([
        ('time_average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', MLPClassifier(max_iter=1000))]
    )

    pipe_parameters = {}

    return pipeline, pipe_parameters


def __perceptron():
    pipeline = Pipeline([
        ('time_average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', Perceptron())]
    )

    pipe_parameters = {}

    return pipeline, pipe_parameters


def __svc():
    pipeline = Pipeline([
        ('vectorization', Vectorizer()),
        ('scaling', StandardScaler()),
        ('estimator', SVC())]
    )

    pipe_parameters = {
        'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'estimator__C': [1, 10],
        'estimator__gamma': ['scale', 'auto'],
        'estimator__shrinking': [False, True],
        'estimator__probability': [False, True]
    }

    #pipe_parameters = {}

    return pipeline, pipe_parameters


def __linear_svc():
    pipeline = Pipeline([
        ('average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', LinearSVC(max_iter=3000))]
    )

    pipe_parameters = {
        'estimator__penalty': ['l1', 'l2'],
        'estimator__loss': ['hinge', 'squared_hinge'],
        'estimator__dual': ['auto', False, True],
        'estimator__multi_class': ['ovr', 'crammer_singer'],
        'estimator__C': [0.5, 1, 5, 10]
    }

    # pipe_parameters = {}

    return pipeline, pipe_parameters


def __vectorized_logistic_regression():
    pipeline = Pipeline([
        ('vectorization', Vectorizer()),
        ('scaling', StandardScaler()),
        ('estimator', LogisticRegression())]
    )

    pipe_parameters = {
        'estimator__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        'estimator__penalty': ['l1', 'l2', 'elasticnet', None],
        'estimator__C': [1, 10],
        'estimator__multi_class': ['auto', 'ovr', 'multinomial'],
        'estimator__dual': [True],
        'estimator__l1_ratio': [None, 0.5],
        'estimator__max_iter': [200]
    }

    # pipe_parameters = {}

    return pipeline, pipe_parameters


def __gaussian_naive_bayes():
    pipeline = Pipeline([
        ('average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', GaussianNB())]
    )

    pipe_parameters = {}

    return pipeline, pipe_parameters


#def __multinomial_naive_bayes():
#    pipeline = Pipeline([
#        ('average', sklearn_custom_steps.TimeAverage()),
#        ('scaling', StandardScaler()),
#        ('estimator', MultinomialNB())]
#    )
#
#    pipe_parameters = {}
#
#    return pipeline, pipe_parameters


def __decision_tree():
    pipeline = Pipeline([
        ('average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', DecisionTreeClassifier())]
    )

    pipe_parameters = {}

    return pipeline, pipe_parameters


def __gradient_boost():
    pipeline = Pipeline([
        ('average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', GradientBoostingClassifier())]
    )

    pipe_parameters = {}

    return pipeline, pipe_parameters


def __xgbc():
    pipeline = Pipeline([
        ('average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', XGBClassifier())]
    )

    pipe_parameters = {}

    return pipeline, pipe_parameters


def __lgbm():
    pipeline = Pipeline([
        ('average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', LGBMClassifier())]
    )

    pipe_parameters = {}

    return pipeline, pipe_parameters


def __knn():
    pipeline = Pipeline([
        ('average', sklearn_custom_steps.TimeAverage()),
        ('scaling', StandardScaler()),
        ('estimator', KNeighborsClassifier())]
    )

    pipe_parameters = {}

    return pipeline, pipe_parameters


def __xgboost_grid_search():
    pipeline = Pipeline([
        ('vectorization', Vectorizer()),
        ('scaling', StandardScaler()),
        ('estimator', XGBClassifier(
            learning_rate=0.02,
            n_estimators=600,
            objective='binary:logistic',
            silent=True,
            nthread=1))]
    )

    pipe_parameters = {
        'estimator__alpha': [0.5, 1, 5, 10],
        'estimator__selection': ['cyclic', 'random'],
        'estimator__positive': [False, True],
        'estimator__fit_intercept': [False, True]
    }

    #pipe_parameters = {}

    return pipeline, pipe_parameters


def __pca_logistic_regression():
    pipeline = Pipeline([
        ('time_average', sklearn_custom_steps.TimeAverage()),
        ('pca', KernelPCA(n_components=50)),
        ('scaling', StandardScaler()),
        ('estimator', LogisticRegression(max_iter=2000))
    ]
    )

    pipe_parameter = {}

    return pipeline, pipe_parameter


def __run_grid_search(gs_estimators, x, y):
    fitted_estimators = list()
    for gs_estimator in gs_estimators:
        gs_estimator.fit(x, y)
        fitted_estimators.append(gs_estimator)

    return fitted_estimators


if __name__ == '__main__':

    fs = [(obj, name) for name, obj in inspect.getmembers(sys.modules[__name__])
          if (inspect.isfunction(obj)
              and name.startswith('__') and 'svc' in name)]

    fs = (
        (__logistic_regression, 'normal lr'),
        #(__gradient_boost, 'gradient boosting'),
        #(__ridge_cv, 'ridge')
    )

    protocol = protocol_dto.LOCA_LEC
    protocol_name = protocol.name
    protocol_directory = protocol.base_directory
    condition = ''
    subjects = protocol.subjects
    classes = protocol.classes_folder_names

    # To crop the data
    start_sample, end_sample = protocol.start_end_time_sample

    file_path = "grid_search_report_" + protocol_name + ".txt"
    f = open(file_path, "w")

    for subject in subjects:
        print(subject)
        f.write(subject + '\n')

        data_directory = os.path.join(protocol_directory, subject, condition)

        x, y, labels = \
            load_x_y.load_x_y_labels_from_brainstorm_folder(
                data_directory, class_folder_names=classes, verbose=False, with_channel_indexes=False)

        x = process_data.crop_frames(x, time_sample_1=start_sample, time_sample_2=end_sample)

        for function, function_name in fs:
            msg = f'{function_name}: '
            f.write(msg)

            pipeline, pipeline_parameters = function()

            gs_cv = \
                GridSearchCV(
                    estimator=pipeline, param_grid=pipeline_parameters, cv=5,
                    scoring='accuracy', refit=True, verbose=0)
            gs_cv.fit(x, y)

            msg = f'{gs_cv.best_score_}'
            f.write(msg + '\n')

            msg = f'{gs_cv.best_params_}'
            f.write(msg + '\n')

    f.close()

    print(file_path)
