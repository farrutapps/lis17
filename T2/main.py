#TODO names (input_data, input_data_test etc),feature selection, change this to class,

#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import csv_manager
from features import Feature
import cross_validation as cv

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
y_source = data_train[:,1].reshape(n_samples,n_dimensions_y)
x_source = data_train[:,n_dimensions_y+1:].reshape(n_samples,n_dimensions_x)


# #####Feature transform
# constant = True
# first = False
# second = False
# exponential = False
#
#
# # feature_vec contains approved features
# feature_vec = []
#
# if constant:
#     feature_vec.append( Feature(np.array([]),'multiply') )
#
# if first:
#     for i in range(n_dimensions_x):
#         feature_vec.append( Feature(np.array([i]),'multiply') )
#
# if second:
#     for i in range(n_dimensions_x):
#         for j in range(n_dimensions_x):
#             feature_vec.append( Feature(np.array([i,j]),'multiply') )
#
# if exponential:
#     for i in range(n_dimensions_x):
#         feature_vec.append( Feature(np.array([i]),'exp') )
#
# for k in range(K):
#     print("Selecting feature {} \t(status: {} %) ".format(k,round(100*float(k)/float(K),1)))
#     # select feature from feature_vec_pool with smalles RMSE and stack it to feature_vec
#     rmse = np.zeros( len(feature_vec_pool) )
#     for i in range( len(feature_vec_pool) ):
#         feature_i = feature_vec
#         feature_i.append( feature_vec_pool[i] )
#
# # Transform samples
# if constant or first or second or exponential:
#     x_source_tf = feature_transform(feature_i,x_source)
#
# else:
#     x_source_tf = x_source
#
# # fit data
# data_cv = np.hstack((ids,y_source,x_source_tf))
# cross_validation = cv.CrossValidation(data_cv,5)
# lin_reg = lr.LinearRegression()
# rmse = cross_validation.start_cross_validation(lin_reg)
# print("RMSE = {}".format(i,rmse))


"""
Predict output with chosen features and learned coefficients beta
"""
# # load test data and transform samples
# data_test = data_loader.restore_from_file('test.csv')
# n_samples_test = data_test.shape[0]
# ids_test = data_test[:,0].reshape(n_samples_test,1)
# x_test = data_test[:,1:].reshape(n_samples_test,n_dimensions_x)
# x_test_tf = feature_transform(feature_vec_pool,x_test)

# # predict output
# y_test = lm.predict(x_test_tf)

# #save output
# header = np.array(['Id','y']).reshape(1,2)
# dump_data = np.hstack((ids_test,y_test))
# data_loader.save_to_file('results.csv',dump_data,header)
