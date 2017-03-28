"""
Comments Sam:
- Is ProgrammaticError used? ==> raise ValueError or general helper class?
- Feature: F(x), w/o y, is it? Solver F(x)*beta = y has to be rewritten as F(x,y)=0
- dimension discussion: (n,) or (n,1)
- indent: google or python standard?

"""

import numpy as np

class ProgrammaticError(Exception):
    """Exception raised when method gets called at a wrong time instance.

    Attributes:
    msg  -- The error message to be displayed.
    """

    def __init__(self, msg):
        self.msg = msg
        print("\033[91mERROR: \x1b[0m {}".format(msg))

class LinearRegression:
    """
    Class for linear regression model.
    """
    # Members
    # Vector in equation: y = F(x,y) * beta, where F(x,y) is the feature vector.
    beta = None
    fitting_done = False

    def __init__(self):
        self.fitting_done = False

    ### fit:
    ## arguments:
    # x: np.array of dimensions (n,f). n number of samples, f number of features. containing the data to be fitted. IT NEEDS TO BE TRANSFORMED ALREADY (in case of non-linear curve fits).
    # y: np.array of dimensions (n,)
    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError("y.shape must be (n,). x.shape must be (n,f)")

        # Least squares parameter estimation. np.linal.pinv calculates psdeudo inverse: inv(X' * X)*X'
        self.beta = np.dot(np.linalg.pinv(x), y)
        self.fitting_done = True

    def predict(self,x):
        if not self.fitting_done:
            # raise Value error? 
            raise ProgrammaticError("Before you use the model for query, you have "
                              "to set the feature vector and fit it.")
        return np.dot(x, self.beta)

    ##returns mean squared error
    def validate(self, x_validate, y_validate):
        return np.mean(self.error_function(self.predict(x_validate),y_validate))

    def error_function(self, predictions, target_values):
        return (predictions - target_values)**2
