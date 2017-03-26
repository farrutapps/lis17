#TODO names (input_data, input_data_test etc),feature selection, change this to class,

#!/usr/bin/env python
import numpy as np
import csv_manager
import Features as features
import linear_regression as lin_reg
import matplotlib.pyplot as plt

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")



# loading training data
data_loader = csv_manager.CsvManager('data')
data_train = data_loader.restore_from_file('train.csv')
n_samples = training_data.shape[0]
n_dimensions_x = 15
n_dimensions_y = 1
ids = data_train[:,0].reshape(n_samples,1)
y_train = data_train[:,1].reshape(n_samples,n_dimensions_y)
x_train = data_train[:,n_dimensions_y+1:].reshape(n_samples,n_dimensions_x)


# Split data into training and validation
ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = input_data[:idx_switch, :]
training_output = output_data[:idx_switch, :]
validation_input = input_data[idx_switch:, :]
validation_output = output_data[idx_switch:, :]

#Feature Vector
feature_vec = [features.LinearX1(), features.Identity()] ## TODO: Select model

# Compute feature matrix
n_features = len(self.feature_vec)
	x_transformed = np.zeros([n_samples, n_features])
    for i in range(n_samples):
      for j in range(n_features):
        x_transformed[i, j] = self.feature_vec[j].evaluate(x)



# Linear Regression
lm = lin_reg.LinearRegression()
lm.fit(training_input, training_output)


# Validation
rmse = lm.validate(validation_input, validation_output)**0.5
print('RMSE: {}'.format(rmse))
print(' ')
print('feature weights \n{}'.format(lm.beta))


# load test data
test_data = data_loader.restore_from_file('test.csv')
n_samples_test = test_data.shape[0]
test_id = test_data[:,0].reshape(n_samples_test,1)
test_input_data = test_data[:,1:].reshape(n_samples_test,n_dimensions_input)


# predict output
test_output = lm.predict(test_input_data)


#save output
header = np.array(['id','y']).reshape(1,2)
dump_data = np.hstack((test_id,test_output))
data_loader.save_to_file('sample.csv',dump_data,header)

