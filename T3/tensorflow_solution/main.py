from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import urllib

import numpy as np
import tensorflow as tf

import csv_manager as csv
tf.logging.set_verbosity(tf.logging.INFO)

def compile_training_csv(filename_source, split_ratio):
    print('compile training csv')

    if split_ratio > 1 or split_ratio < 0:
        raise ValueError('ERROR: ratio must be: 0 <= ratio <= 1')

    # read from file
    man = csv.CsvManager('../data')
    data_source = man.restore_from_hdf5(filename_source)

    #shuffle data
    np.random.shuffle(data_source)

    n_data = data_source.shape[0]

    # put y column at the end
    data = data_source[:, 2:].astype(object)
    target = data_source[:, 1].reshape(n_data, 1)
    target = target.astype(int)
    target = target.astype(object)
    print (type(target[0, 0]))

    data_source = np.hstack([data,target])
    print(type(data_source[5,5]))
    print(type(data_source[0,-1]))

    print(data_source.shape)
    num_columns = data_source.shape[1]

    # split into test and training set
    split_index = int(n_data*split_ratio)

    csv_train = data_source[split_index:,:]

    csv_validate = data_source[0:split_index,:]

    # define headers according to tensorflow doc
    n_validate = int(n_data*split_ratio)
    n_train = n_data - n_validate

    column_headers = ['x{}'.format(i) for i in range(num_columns)]
    column_headers[0] = str(n_train)
    column_headers[1] = num_columns - 1
    column_headers[-1] = 'y'

    column_headers = np.array(column_headers).reshape(1, num_columns)

    types = '%f,' * 100 + '%i'
    man.save_to_file('train.csv', csv_train, column_headers, types=types)

    column_headers[0,0] = str(n_validate)

    man.save_to_file('validate.csv', csv_validate, column_headers, types=types)

# Data sets
TRAINING_SOURCE = 'train.h5'
TRAINING = "train.csv"

VALIDATE = 'validate.csv'

DIRECTORY = '../data/'

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(DIRECTORY + TRAINING) or not os.path.exists(DIRECTORY + VALIDATE):
    compile_training_csv(TRAINING_SOURCE, 0.2)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=DIRECTORY + TRAINING,
      target_dtype=np.int,
      features_dtype=np.float)
  validation_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=DIRECTORY + VALIDATE,
      target_dtype=np.int,
      features_dtype=np.float)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=100)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[100],
                                              n_classes=5,
                                              model_dir="/tmp/T3_model_new15")
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

  # Fit model.
  classifier.fit(input_fn=get_train_inputs, steps=2000)

  # Define the test inputs
  def get_validation_inputs():
    x = tf.constant(validation_set.data)
    y = tf.constant(validation_set.target)

    return x, y

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_validation_inputs,
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify test data
  print('classify test data')
  man = csv.CsvManager('../data/')
  x_test = man.restore_from_hdf5('test.h5')
  n_data = x_test.shape[0]
  ids = x_test[:,0].reshape(n_data,1)

  predictions = np.array(list(classifier.predict(x_test[:,1:], as_iterable=True))).reshape(n_data,1)

  prediction_output = np.hstack([ids, predictions])

  header = np.array(['Id', 'y']).reshape(1, 2)
  man.save_to_file('results.csv', prediction_output, header)

if __name__ == "__main__":
    main()