import numpy as np
from sklearn.kernel_ridge import KernelRidge


class KernelRidgeRegression():
    ""
    # This class implements the sklearn tool Ridge Regression in a way that it works with the class CrossValidation.
    ""

    lambda_ = None
    kernel_ridge = None

    def __init__(self, lambda_, kernel='linear', gamma=None, degree=3):
        # See definition of Ridge Regression const function.
        self.lambda_ = lambda_
        self.gamma = gamma

        self.degree = degree
        # Use strings to choose kerenl: 'linear', 'rbf', see scikit documentation
        self.kernel = kernel
        self.kernel_ridge = KernelRidge(alpha=lambda_, gamma = self.gamma, kernel=self.kernel, degree=self.degree)

    def fit(self, x, y):
        self.kernel_ridge.fit(x, y)

    def validate(self, x, y):
        return np.mean(self.error_function(self.kernel_ridge.predict(x), y))

    def error_function(self, predictions, target_values):
        return (predictions - target_values) ** 2

    def predict(self, x):
        return self.kernel_ridge.predict(x)
