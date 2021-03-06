# TODO names (input_data, input_data_test etc),feature selection, change this to class,

# !/usr/bin/env python
import numpy as np
import csv_manager
from features import Feature
import scikit_regression as sci_reg
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import os

import cross_validation as cv

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings

warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")


class Parameter():
    def __init__(self, alpha, h=None):
        self.alpha = alpha
        self.h = h
        self.rmse = None
        self.coef = None


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


"""
Find model and corresponding coefficients beta
"""
# loading training data
data_loader = csv_manager.CsvManager('data')
data_train = data_loader.restore_from_file('train.csv')

# preprocess data

# outlier rejection.
outlier_rejection = False

if (outlier_rejection):
    print 'Doing outlier rejection.'
    # reject datapoints where the following is not true: a < norm(x) > b
    a = 3.5
    b = 7
    print 'a={}, b={}'.format(a, b)
    i = 0
    delete_index = []
    for row in data_train:
        norm = np.linalg.norm(row[2:])
        print norm
        if norm < a or norm > b:
            delete_index.append(i)

        i += 1

    data_train = np.delete(data_train, delete_index, axis=0)
    print data_train.shape

n_samples = data_train.shape[0]
n_dimensions_x = 15
n_dimensions_y = 1

# shuffle order of samples. This makes sure that training and validation sets contain random samples.
np.random.shuffle(data_train)
ids = data_train[:, 0].reshape(n_samples, 1)
y_source = data_train[:, 1].reshape(n_samples, n_dimensions_y)
x_source = data_train[:, n_dimensions_y + 1:].reshape(n_samples, n_dimensions_x)

# Feature transform
transform = True
constant = True
first = True
second = True
third = True
exponential = False

feature_vec = []

if transform:
    # constant
    if constant:
        feature_vec = [Feature(np.array([]), 'multiply')]

    # first order monomials: linear (additional features: 15, total 16)

    if first:
        for i in range(n_dimensions_x):
            feature_vec.append(Feature(np.array([i]), 'multiply'))

    if second:
        # second order monomials: quadratic (additional features: 15*15 = 225, total 241)
        for i in range(n_dimensions_x):
            for j in range(n_dimensions_x):
                feature_vec.append(Feature(np.array([i, j]), 'multiply'))

    if third:
        for i in range(n_dimensions_x):
            for j in range(n_dimensions_x):
                for k in range(n_dimensions_x):
                    feature_vec.append(Feature(np.array([i, j, k]), 'multiply'))

    if exponential:
        # exponential function and logarithmic function
        for i in range(n_dimensions_x):
            feature_vec.append(Feature(np.array([i]), 'exp'))
            # feature_vec.append( Feature(np.array([i]),'log')  # why is log not working?

    # Transform samples
    if constant or first or second or third or exponential:
        print 'transform features'

        if constant and first and second and third and not exponential and os.path.isfile('data/up_to_three.csv'):
            print 'load transform from file'
            # data_loader.save_to_file('up_to_three.csv',np.hstack((y_source,x_source_tf)),np.array(['h']*(x_source_tf.shape[1]+1)).reshape(1,x_source_tf.shape[1]+1))
            source_tf = data_loader.restore_from_file('up_to_three.csv')
            x_source_tf = source_tf[:, 1:]
            y_source = source_tf[:, 0].reshape(source_tf.shape[0], 1)

        else:
            x_source_tf = feature_transform(feature_vec, x_source)
    else:
        x_source_tf = x_source

else:
    x_source_tf = x_source

print ids.shape
print y_source.shape
print x_source_tf.shape
data_cv = np.hstack((ids, y_source, x_source_tf))

cross_validation = cv.CrossValidation(data_cv, int(20))

# lin_reg = lr.LinearRegression()

# cross validate over lamda in ridge regression
cross_validate = False

if cross_validate:
    print 'Doing Cross Validation'

    results = []
    scale = 0.02
    scale_h = 1.
    for i in range(10, 20, 1):
        print i
        for j in [1]:
            param = Parameter(alpha=i * scale)

            ridge = Lasso(param.alpha)
            reg = sci_reg.ScikitRegression(ridge)

            error = []
            for k in [1]:
                error.append(cross_validation.start_cross_validation(reg))

            param.coef = ridge.coef_
            param.rmse = np.mean(error)
            results.append(param)

    best = [p_best for p_best in results if p_best.rmse == min([p.rmse for p in results])]
    best = best[0]
    print 'best rmse: {} alpha = {}'.format(best.rmse, best.alpha)

    # count zero entries
    print best.coef
    number_non_zero = np.linalg.norm(best.coef, 0)
    print 'number of non zero entires {}/{}'.format(number_non_zero, best.coef.shape[0])


    # kill features with zero entries.
    # delete_index = []
    #
    # for i in range(best.coef.shape[0]):
    #     if best.coef[i] == 0.:
    #         delete_index.append(i)
    #
    # x_lasso = np.delete(x_source_tf,delete_index,0)
    # y_lasso = np.delete(y_source, delete_index, 0)
    # print x_lasso.shape
    # cv_reduced = cv.CrossValidation(np.hstack((y_lasso,y_lasso,x_lasso)),20)
    #
    # results_reduced = []
    # scale_red = 1
    # for i in range(750,770,1):
    #     print i
    #     param_reduced = Parameter(i*scale_red,0)
    #     ridge_reduced = Ridge(param_reduced.alpha)
    #     ridge_reduced_holder = sci_reg.ScikitRegression(ridge_reduced)
    #
    #     param_reduced.rmse = cv_reduced.start_cross_validation(ridge_reduced_holder)
    #     results_reduced.append(param_reduced)
    #
    # best_reduced = [p_best for p_best in results_reduced if p_best.rmse == min([p.rmse for p in results_reduced])]
    # best_reduced = best_reduced[0]
    # print 'best rmse for reduced problem: {}. alpha = {}'.format(best_reduced.rmse,best_reduced.alpha)



# End Cross validation

else:
    print 'Making test prediction'

    reg = Lasso(0.28)
    reg.fit(x_source_tf,y_source)
    """
    Predict output with chosen features and learned coefficients beta
    """
    # load test data and transform samples
    data_test = data_loader.restore_from_file('test.csv')
    n_samples_test = data_test.shape[0]
    ids_test = data_test[:, 0].reshape(n_samples_test, 1)
    x_test = data_test[:, 1:].reshape(n_samples_test, n_dimensions_x)
    x_test_tf = feature_transform(feature_vec, x_test)


    # predict output
    if transform:
        y_test = reg.predict(x_test_tf).reshape(n_samples_test, 1)

    else:
        y_test = reg.predict(x_test).reshape(n_samples_test, 1)

    # save output
    header = np.array(['Id', 'y']).reshape(1, 2)
    dump_data = np.hstack((ids_test, y_test))
    data_loader.save_to_file('results.csv', dump_data, header)
