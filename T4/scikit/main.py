import numpy as np
import csv_manager
from features import Feature
from parameter_manager import ParameterManager
import scikit_model as sci_holder
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

import cross_validation as cv


class Parameter():
    def __init__(self, alpha):
        self.alpha = alpha
        self.accuracy = None

"""
Methods
"""
# transforms data into feature matrix st. non-linear features can be computed using linear regression
# arguments:
#   x, np.array of dimension (n,), n number of samples
# outputs:
#   x_tf, np.array of dimension (n, f), n number of samples, f number of features.

def feature_transform(feature_vec, x):
    n_features = len(feature_vec)
    n_samples = x.shape[0]

    x_tf = np.zeros([n_samples, n_features])
    for i in range(n_samples):
        for j in range(n_features):
            x_tf[i, j] = feature_vec[j].evaluate(x[i, :])

    return x_tf


def compute_feature_vec(orders):
    feature_vec = []

    for element in orders:
        if element > 3 or element < 0:
            raise ValueError('Can only create monomials of degree 0 to 3')

    # constant
    if 0 in orders:
        feature_vec = [Feature(np.array([]), 'multiply')]

    # first order monomials: linear (additional features: 15, total 16)

    if 1 in orders:
        for i in range(n_dimensions_x):
            feature_vec.append(Feature(np.array([i]), 'multiply'))

    if 2 in orders:
        # second order monomials: quadratic (additional features: 15*15 = 225, total 241)
        for i in range(n_dimensions_x):
            for j in range(n_dimensions_x):
                feature_vec.append(Feature(np.array([i, j]), 'multiply'))

    if 3 in orders:
        for i in range(n_dimensions_x):
            for j in range(n_dimensions_x):
                for k in range(n_dimensions_x):
                    feature_vec.append(Feature(np.array([i, j, k]), 'multiply'))

    return feature_vec

    # if exponential:
    #     # exponential function and logarithmic function
    #     for i in range(n_dimensions_x):
    #         feature_vec.append(Feature(np.array([i]), 'exp'))
    #         # feature_vec.append( Feature(np.array([i]),'log')  # why is log not working?


"""
Model fitting
"""
# load data
data_loader = csv_manager.CsvManager('../data')
#data_train = data_loader.restore_from_file('train.csv')

data_labeled = pd.read_hdf("../data/train_labeled.h5", "train")
data_unlabeled = pd.read_hdf("../data/train_unlabeled.h5", "train")
data_train_labeled = data_labeled.as_matrix()
data_train_unlabeled = data_unlabeled.as_matrix()

print 'shape of   labeled data set: {}'.format(data_train_labeled.shape)
print 'shape of unlabeled data set: {}'.format(data_train_unlabeled.shape)
n_samples_labeled = data_train_labeled.shape[0]
n_samples_unlabeled = data_train_unlabeled.shape[0]
n_dimensions_x = 128
n_dimensions_y = 1

# shuffle order of samples. This makes sure that training and validation sets contain random samples.
np.random.shuffle(data_train_labeled)
np.random.shuffle(data_train_unlabeled)

ids = data_labeled.index.values.reshape(n_samples_labeled,1)

y_source = data_train_labeled[:, 0].reshape(n_samples_labeled, n_dimensions_y)
x_source = data_train_labeled[:, 1:].reshape(n_samples_labeled, n_dimensions_x)

# compute feature vector
feature_vec = compute_feature_vec([])

# if feature_vec not empty, do transform.
if feature_vec:
    print 'transform features'
    transformed = True
    x_source_tf = feature_transform(feature_vec, x_source)

else:
    x_source_tf = x_source
    transformed = False


# Scale data
scaler = StandardScaler()

scale_data = True
if scale_data:
    print 'scale data'
    scaler.fit(x_source_tf)
    x_source_tf = scaler.transform(x_source_tf)

# Cross validation
data_cv = np.hstack((ids, y_source, x_source_tf))
cross_validation = cv.CrossValidation(data_cv, 3)

# switch on/off if cross validation or test data prediction
cross_validate = False

if cross_validate:
    print 'Doing Cross Validation (not adapted yet)'

    # param_manager = ParameterManager()
    #
    # myrange = [5*x for x in range(1,5)]
    #
    # # Define parameters here: (parameter_name, [parameter_values])
    # parameter_settings = [('alpha', myrange), ('layer_size', [(100,100,100,100,100,100,100,100), (50)]),
    #                       ('solver',('lbfgs','adam'))]
    #
    # # Compute all diferent possible parameter_sets
    # param_manager.compute_parameter_sets(parameter_settings)
    #
    # # loop through parameter sets and start computation
    # for ps in param_manager.parameter_sets:
    #     print 'progress: {}/{}'.format(param_manager.parameter_sets.index(ps), len(param_manager.parameter_sets))
    #
    #     classifier = MLPClassifier(solver=ps['solver'], alpha=ps['alpha'], hidden_layer_sizes=ps['layer_size'], warm_start=True)
    #
    #     holder = sci_holder.ScikitModel(classifier)
    #
    #     # start learning
    #     ps['accuracy'] = cross_validation.start_cross_validation(holder)
    #
    #     # print results
    #     print ps
    #
    # #find best result
    # accuracies = [p['accuracy'] for p in param_manager.parameter_sets]
    # best = [p for p in param_manager.parameter_sets if p['accuracy'] == max(accuracies)]
    # print best
    #
    # print 'standard deviation of results: {}'.format(np.std(accuracies))


# End Cross validation

else:
    """
        Predict output on test data
    """
    print 'Making test prediction'

    classifier = MLPClassifier(solver='lbfgs', alpha=40, hidden_layer_sizes=(2100,1300))
    classifier.fit(x_source_tf, y_source.reshape(n_samples_labeled))


    # load test data and transform samples
    data_pandas = pd.read_hdf("../data/test.h5", "test")
    data_test = data_pandas.as_matrix()
    n_samples_test = data_test.shape[0]
    ids_test = data_pandas.index.values.reshape(n_samples_test,1)

    x_test = data_test.reshape(n_samples_test, n_dimensions_x)

    # predict output
    if transformed:
        print 'transform data'
        x_test_tf = feature_transform(feature_vec, x_test)

        if scale_data:
            print 'scale data'
            x_test_tf = scaler.transform(x_test_tf)


        y_test = classifier.predict(x_test_tf).reshape(n_samples_test, 1)

    elif scale_data:
        print 'scale data'
        x_test_tf = scaler.transform(x_test)
        y_test = classifier.predict(x_test_tf).reshape(n_samples_test, 1)

    else:
        y_test = classifier.predict(x_test).reshape(n_samples_test, 1)

    # save output
    header = np.array(['Id', 'y']).reshape(1, 2)
    dump_data = np.hstack((ids_test, y_test))
    data_loader.save_to_file('results.csv', dump_data, header)
