#TODO names (input_data, input_data_test etc),feature selection, change this to class,

#!/usr/bin/env python
import numpy as np
import csv_manager
from features import Feature
import lin_reg as lr
import ridge_reg as rr
import matplotlib.pyplot as plt
import cross_validation as cv

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
Find model and corresponding coefficients beta
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


#####Feature transform
constant = True
first = False
second = False
exponential = True


### greedy forward feature selection
# feature_vec_pool contains all possible features
# feature_vec contains approved features
feature_vec_pool = []
feature_vec = []

if constant:
    feature_vec_pool.append( Feature(np.array([]),'multiply') )

if first:
    for i in range(n_dimensions_x):
        feature_vec_pool.append( Feature(np.array([i]),'multiply') )

if second:
    for i in range(n_dimensions_x):
        for j in range(n_dimensions_x):
            feature_vec_pool.append( Feature(np.array([i,j]),'multiply') )

if exponential:
    for i in range(n_dimensions_x):
        feature_vec_pool.append( Feature(np.array([i]),'exp') )

K = 2 # select K most promising features
RMSE = np.zeros(K)
for k in range(K):
    print("Selecting feature {} \t(status: {} %) ".format(k,round(100*float(k)/float(K),1)))
    # select feature from feature_vec_pool with smalles RMSE and stack it to feature_vec
    rmse = np.zeros( len(feature_vec_pool) )
    for i in range( len(feature_vec_pool) ):
        feature_i = feature_vec
        feature_i.append( feature_vec_pool[i] )
        # Transform samples
        if constant or first or second or exponential:
            x_source_tf = feature_transform(feature_i,x_source)

        else:
            x_source_tf = x_source

        data_cv = np.hstack((ids,y_source,x_source_tf))
        cross_validation = cv.CrossValidation(data_cv,5)
        lin_reg = lr.LinearRegression()
        # rmse[i] = cross_validation.start_single_validation(lin_reg)
        rmse[i] = cross_validation.start_cross_validation(lin_reg)
        print("\tfeature {}: \tRMSE = {}".format(i,rmse[i]))

    # take most promising feature: add to feature_vec and delete it from feature_vec_pool
    idx_min = np.argmin(rmse)
    # print feature_vec
    feature_vec.append(feature_vec_pool[idx_min])
    # print feature_vec

    del feature_vec_pool[idx_min]
    RMSE[k] = rmse[idx_min]
    print("\tRMSE = {}".format(RMSE[k]))

plot_err_vs_features = False
if plot_err_vs_features:
    plt.figure()
    plt.plot(range(K),RMSE)
    plt.xlabel('number of features')
    plt.ylabel('RMSE (CV, k=10)')
    plt.show(block=True)

print("RMSE = \n{}\n".format(RMSE.reshape(K,1)))

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
