#TODO names (input_data, input_data_test etc),feature selection, change this to class,

#!/usr/bin/env python
import numpy as np
import csv_manager
import Features as features
import LinearRegressionModel as model
import matplotlib.pyplot as plt

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")



# loading training data
data_loader = csv_manager.CsvManager('data')
training_data = data_loader.restore_from_file('train.csv')
n_samples = training_data.shape[0]
n_dimensions_input = 15
n_dimensions_output = 1
id_data = training_data[:,0].reshape(n_samples,1)
output_data = training_data[:,1:n_dimensions_output+1].reshape(n_samples,n_dimensions_output)
input_data = training_data[:,n_dimensions_output+1:].reshape(n_samples,n_dimensions_input)


# Split data into training and validation
ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = input_data[:idx_switch, :]
training_output = output_data[:idx_switch, :]
validation_input = input_data[idx_switch:, :]
validation_output = output_data[idx_switch:, :]


# Fit model
lm = model.LinearRegressionModel()
lm.set_feature_vector([features.LinearX1(), features.Identity()])
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

