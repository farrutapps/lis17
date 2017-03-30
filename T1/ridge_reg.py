import numpy as np
from sklearn.linear_model import Ridge

class RidgeRegression():
	""
	#This class implements the sklearn tool Ridge Regression in a way that it works with the class CrossValidation.
	""

	lambda_ = None
	ridge = None

	def __init__(self, lambda_):
		# See definition of Ridge Regression const function.
		self.lambda_ = lambda_

		self.ridge = Ridge(alpha = lambda_)

	def fit(self,x,y):
		self.ridge.fit(x,y)

	def validate(self,x,y):
		return np.mean(self.error_function(self.ridge.predict(x),y))

	def error_function(self, predictions, target_values):
		return (predictions - target_values)**2	

	def predict(self,x):
		return self.ridge.predict()

