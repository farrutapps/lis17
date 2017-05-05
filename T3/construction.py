from __future__ import print_function
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
import os
import pandas as pd

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer((None, 100),input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


# def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
#                      drop_hidden=.5):
#     # By default, this creates the same network as `build_mlp`, but it can be
#     # customized with respect to the number and size of hidden layers. This
#     # mostly showcases how creating a network in Python code can be a lot more
#     # flexible than a configuration file. Note that to make the code easier,
#     # all the layers are just called `network` -- there is no need to give them
#     # different names if all we return is the last one we created anyway; we
#     # just used different names above for clarity.
#
#     # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
#     network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
#                                         input_var=input_var)
#     if drop_input:
#         network = lasagne.layers.dropout(network, p=drop_input)
#     # Hidden layers and dropout:
#     nonlin = lasagne.nonlinearities.rectify
#     for _ in range(depth):
#         network = lasagne.layers.DenseLayer(
#                 network, width, nonlinearity=nonlin)
#         if drop_hidden:
#             network = lasagne.layers.dropout(network, p=drop_hidden)
#     # Output layer:
#     softmax = lasagne.nonlinearities.softmax
#     network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
#     return network

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


"""
main function
"""
# control Parameter
model = 'mlp'
num_epochs = 1
ratio_train_validate = 0.8

# get source data for training
n_dimensions_x = 100
n_dimensions_y = 1
data_pandas = pd.read_hdf("data/train.h5", "train")
data_train = data_pandas.as_matrix()
n_samples = data_train.shape[0]
np.random.shuffle(data_train)
ids = data_pandas.index.values.reshape(n_samples,1)
y_source = data_train[:, 0].reshape(n_samples, n_dimensions_y).astype(np.int32)
x_source = data_train[:, 1:].reshape(n_samples, n_dimensions_x)


# Split sourve into training and validation
idx_switch = int(n_samples * ratio_train_validate)
x_train = x_source[:idx_switch, :]
y_train = y_source[:idx_switch, :]
x_val = x_source[idx_switch:, :]
y_val = y_source[idx_switch:, :]
n_samples_train = y_train.shape[0];
n_samples_val = y_val.shape[0];
y_train = y_train.reshape(n_samples_train)
y_val = y_val.reshape(n_samples_val)


# prepare Theano vectors for optimization lateron
input_var = T.matrix('inputs')
target_var = T.ivector('targets')

# Create neural network model (depending on first command line parameter)
print("Building model and compiling functions...")
if model == 'mlp':
    network = build_mlp(input_var)
elif model.startswith('custom_mlp:'):
    depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
    network = build_custom_mlp(input_var, int(depth), int(width),
                               float(drop_in), float(drop_hid))
# elif model == 'cnn':
#     network = build_cnn(input_var)
else:
    print("Unrecognized model type %r." % model)

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
# We could add some weight decay as well here, see lasagne.regularization.

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

# Compile a third function for prediction the output
pred_fn = theano.function([input_var], test_prediction)

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(x_train, y_train, 10, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(x_val, y_val, 10, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))


# After training, we compute and print the val error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(x_val, y_val, 10, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))

# get data for testing and predict labels on test set
data_pandas = pd.read_hdf("data/test.h5", "test")
data_test = data_pandas.as_matrix()
n_samples_test = data_test.shape[0]
ids_test = data_pandas.index.values.reshape(n_samples_test,1)
x_test = data_test.reshape(n_samples_test, n_dimensions_x)

# y_test = lasagne.layers.get_output(network, x_test)
# predict_y = theano.function([input_var], lasagne.layers.get_output(network))
# y_test = predict_y(x_test)
y_test = pred_fn(x_test)
# y_test = lasagne.layers.get_output(network, deterministic=True)
print("x_test = \n{}".format(x_test))
print("y_test = \n{}".format(y_test))
print("x_test.shape = {}".format(x_test.shape))
print("y_test.shape = {}".format(y_test.shape))
print("y_test[1,1] = {}".format(y_test[1,:]))
# # save output
# header = np.array(['Id', 'y']).reshape(1, 2)
# dump_data = np.hstack((ids_test, y_test))
# data_loader.save_to_file('results.csv', dump_data, header)
