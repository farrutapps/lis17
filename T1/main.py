#TODO names (input_data, input_data_test etc),feature selection, change this to class,

#!/usr/bin/env python
import numpy as np
import csv_manager
from features import Feature
import lin_reg
import matplotlib.pyplot as plt
import sys

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")
# supress .pyc files
sys.dont_write_bytecode = True


# computes feature matrix
def feature_transform(feature_vec, x):
    n_features = len(feature_vec)
    n_samples = x.shape[0]

    x_tf = np.zeros([n_samples, n_features])
    for i in range(n_samples):
        for j in range(n_features):
            x_tf[i, j] = feature_vec[j].evaluate(x[i,:])

    return x_tf

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

# Split data into training and validation
ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
x_train = x_source[:idx_switch, :]
y_train = y_source[:idx_switch, :]
x_validate = x_source[idx_switch:, :]
y_validate = y_source[idx_switch:, :]

#Feature Vector
feature_vec = [Feature(np.array([0,0]),'multiply'),Feature(np.array([14]),'exp')] ## TODO: Select model

# Transform samples
x_train_tf = feature_transform(feature_vec,x_train)
x_validate_tf = feature_transform(feature_vec,x_validate)

# Linear Regression
lm = lin_reg.LinearRegression()

# Compute feature matrix
lm.fit(x_train_tf, y_train)

# Validation
rmse = lm.validate(x_validate_tf, y_validate)**0.5
print('RMSE: {}'.format(rmse))
print(' ')
print('feature weights \n{}'.format(lm.beta))

# load test data and transform samples
data_test = data_loader.restore_from_file('test.csv')
n_samples_test = data_test.shape[0]
ids_test = data_test[:,0].reshape(n_samples_test,1)
x_test = data_test[:,1:].reshape(n_samples_test,n_dimensions_x)
x_test_tf = feature_transform(feature_vec,x_test)

# predict output
y_test = lm.predict(x_test_tf)

#save output
header = np.array(['id','y']).reshape(1,2)
dump_data = np.hstack((ids_test,y_test))
data_loader.save_to_file('results.csv',dump_data,header)
