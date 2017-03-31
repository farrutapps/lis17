import numpy as np

class ScikitRegression():
	""
	#This class implements the sklearn tool Ridge Regression in a way that it works with the class CrossValidation.
	""

	scikit_solver = None

	def __init__(self, scikit_solver):
		# See definition of Ridge Regression const function.
		self.scikit_solver = scikit_solver


	def fit(self,x,y):
		self.scikit_solver.fit(x,y)

	def validate(self,x,y):
		return np.mean(self.error_function(self.scikit_solver.predict(x),y))

	def error_function(self, predictions, target_values):
		return (predictions - target_values)**2

	def predict(self,x):
		return self.scikit_solver.predict(x)

