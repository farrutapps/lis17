import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    ### Class for linear regression

    # Members
    # beta: multiplier of features y = sum beta_i * Phi_i(x)
    beta = None
    fitting_done = False

    def __init__(self):
        self.fitting_done = False
        beta = None


    ### fit: estimate beta for given X, y
    ## arguments:
    # X: transformed feature matrix, X = PHI(x)
    #    np.array of dimensions (n,f). n number of samples, f number of features. containing the data
    #    to be fitted. IT NEEDS TO BE TRANSFORMED ALREADY (in case of non-linear curve fits).
    # y: np.array of dimensions (n,)
    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("y.shape must be (n,). X.shape must be (n,f)")

        # Least squares parameter estimation. np.linal.pinv calculates psdeudo inverse: inv(X' * X)*X'
        self.beta = np.dot(np.linalg.pinv(X), y)
        self.fitting_done = True


    ### validate: calculate error with given beta, X, y
    ## arguments:
    # X: transformed feature matrix, X = PHI(x)
    #    np.array of dimensions (n,f). n number of samples, f number of features. containing the data
    #    to be fitted. IT NEEDS TO BE TRANSFORMED ALREADY (in case of non-linear curve fits).
    # y: np.array of dimensions (n,)    # returns mean square error
    ## additional information:
    # change error model (squared error) in error_function in this class
    def validate(self, X, y):
        return np.mean(self.error_function(self.predict(X), y))


    ### predict:
    ## arguments:
    # X: transformed feature matrix, X = PHI(x)
    #    np.array of dimensions (n,f). n number of samples, f number of features. containing the data
    #    to be fitted. IT NEEDS TO BE TRANSFORMED ALREADY (in case of non-linear curve fits).
    def predict(self, X):
        if not self.fitting_done:
            raise ValueError("Before you use the model for query, you have "
                              "to set the feature vector and fit it.")
        return np.dot(X, self.beta)


    # define error model: quadratic error
    def error_function(self, predictions, target_values):
        return (predictions - target_values)**2


    # helper: get beta used for testing
    def get_beta(self):
        return self.beta


# ### Testing: Sam
# N = 100
# x = np.linspace(0, 1, N)
# beta = np.array([2, 1])
# y_true = beta[0]*x + beta[1]
# y = y_true + 0.1*np.random.randn(x.shape[0]);
# if False:
#     plt.plot(x,y)
#     plt.show()
# X = np.hstack( (x.reshape(N,1), np.ones((N,1)) ))
#
# lm = LinearRegression()
# lm.fit(X, y)
# print("validate with true data (not possible): {}".format(lm.validate(X, y_true)))
# print("predict y=2*0.5 + 1 = {}".format( lm.predict([0.5, 1.0])) )
# print("beta_est = {}".format(lm.get_beta()))
