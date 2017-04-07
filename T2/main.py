#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import csv_manager
from features import Feature
import cross_validation as cv
import k_nearest_neighbor as knn

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")


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
y_source = data_train[:,1]
x_source = data_train[:,n_dimensions_y+1:].reshape(n_samples,n_dimensions_x)


k_neighbors = 10
clf = knn.kNearestNeighbor(k_neighbors)
clf.fit(x_source, y_source)



"""
Predict output with chosen features and learned coefficients beta
"""
# load test data and transform samples
data_test = data_loader.restore_from_file('test.csv')
n_samples_test = data_test.shape[0]
ids_test = data_test[:,0].reshape(n_samples_test,1)
x_test = data_test[:,1:].reshape(n_samples_test,n_dimensions_x)

# predict output
y_test = clf.predict(x_test).reshape(n_samples_test,1)

#save output
header = np.array(['Id','y']).reshape(1,2)
dump_data = np.hstack((ids_test,y_test))
data_loader.save_to_file('results.csv',dump_data,header)
