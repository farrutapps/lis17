import numpy as np

class LinearRegression:
    """
    Class for linear regression model.
    """
    # Members
    beta = None
    fitting_done = False

    def __init__(self):
        self.fitting_done = False
        beta = None

    ### fit:
    ## arguments:
    # x: np.array of dimensions (n,f). n number of samples, f number of features. containing the data
    #    to be fitted. IT NEEDS TO BE TRANSFORMED ALREADY (in case of non-linear curve fits).
    # y: np.array of dimensions (n,)
    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError("y.shape must be (n,). x.shape must be (n,f)")

        # Least squares parameter estimation. np.linal.pinv calculates psdeudo inverse: inv(X' * X)*X'
        self.beta = np.dot(np.linalg.pinv(x), y)
        self.fitting_done = True

    ### predict:
    ## arguments:
    # x: np.array of dimension (n,f). n number of samples, f number of features. containing the data
    #    to be fitted. IT NEEDS TO BE TRANSFORMED ALREADY (in case of non-linear curve fits).
    def predict(self,x):
        if not self.fitting_done:
            raise ValueError("Before you use the model for query, you have "
                              "to set the feature vector and fit it.")
        return np.dot(x, self.beta)

    # returns mean error
    def validate(self, x_validate, y_validate):
        return np.mean(self.error_function(self.predict(x_validate),y_validate))

    # define quadratic error
    def error_function(self, predictions, target_values):
        return (predictions - target_values)**2
