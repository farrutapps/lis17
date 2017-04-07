#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os

import csv_manager
from features import Feature
import cross_validation as cv
import k_nearest_neighbor as knn

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")


"""
Methods
"""
### transforms data into feature matrix st. non-linear features can be computed using linear regression
## arguments:
#   x, np.array of dimension (n,), n number of samples
# outputs:
#   x_tf, np.array of dimension (n, f), n number of samples, f number of features.
def feature_transform(feature_vec, x):
    n_features = len(feature_vec)
    n_samples = x.shape[0]

    x_tf = np.zeros([n_samples, n_features])
    for i in range(n_samples):
        for j in range(n_features):
            x_tf[i, j] = feature_vec[j].evaluate(x[i,:])

    return x_tf

"""
Classification using k nearest neighbors
"""
# loading training data
data_loader = csv_manager.CsvManager('data')
data_train = data_loader.restore_from_file('train.csv')
n_samples = data_train.shape[0]
n_dimensions_x = 15
n_dimensions_y = 1

#shuffle order of samples. This makes sure that training and validation sets contain random samples.
np.random.shuffle(data_train)
ids = data_train[:,0].reshape(n_samples,1)
y_source = data_train[:,1].reshape(n_samples,n_dimensions_y)
x_source = data_train[:,n_dimensions_y+1:].reshape(n_samples,n_dimensions_x)

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





if False:
    print "starting crossvaldation to evaluate k_neighbors for classification..."
    k_neighbors_arr = np.arange(2,20,2)

    for k_neighbors in k_neighbors_arr:

        data_cv = np.hstack((ids,y_source,x_source_tf))
        cross_validation = cv.CrossValidation(data_cv,10)
        k_nn = knn.kNearestNeighbor(k_neighbors)
        acc = cross_validation.start_cross_validation(k_nn)
        print("k_neighbors = {}: \taccuracy = {}".format(k_neighbors,acc))


"""
Predict output using k nearest neighbors
"""
# crossvalidaiotn shows that k_neighbors = 6 is a good choice
k_neighbors = 6
clf = knn.kNearestNeighbor(k_neighbors)
clf.fit(x_source_tf, y_source.ravel())

# load test data and transform samples
data_test = data_loader.restore_from_file('test.csv')
n_samples_test = data_test.shape[0]
ids_test = data_test[:,0].reshape(n_samples_test,1)
x_test = data_test[:,1:].reshape(n_samples_test,n_dimensions_x)
x_test_tf = feature_transform(feature_vec, x_test)

# predict output
y_test = clf.predict(x_test_tf).reshape(n_samples_test,1)

#save output
header = np.array(['Id','y']).reshape(1,2)
dump_data = np.hstack((ids_test,y_test))
data_loader.save_to_file('results.csv',dump_data,header)
