import numpy as np
from sklearn.metrics import accuracy_score


class ScikitModel():
	""
	# This class serves as a interface between the CrossValidation class and a lot of solvers form the scikit sklearn library.

	## USAGE:
	# 1) initialize scikit solver, such as sklearn.linear_model.Ridge with desired parameters.
	# 2) Give the instance of the solver from 1) as an argument to this class.
	# 3) Hand the instance of this class to the CrossValidation class.
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
		return accuracy_score(target_values, predictions)

	def predict(self,x):
		return self.scikit_solver.predict(x)

