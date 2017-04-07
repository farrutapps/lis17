import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score


class kNearestNeighbor():
	"""
	This class implements the sklearn tool k nearest neigbour classification in a way that it works with the class CrossValidation.
	"""

	k_neighbors = None
	classification = None

	def __init__(self, k_neighbors):
		self.k_neighbors = k_neighbors
		self.classification = neighbors.KNeighborsClassifier(k_neighbors, weights='distance')

	def fit(self,x,y):
		self.classification.fit(x,y)

	def validate(self,x,y):
		return np.mean(self.error_function(self.classification.predict(x),y))

	def error_function(self, predictions, target_values):
		return accuracy_score(predictions, target_values)

	def predict(self,x):
		return self.classification.predict(x)


# ### Testing Sam
# x = np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]])
# y = np.array([1,1,1,0,1,1,0,0,1])
# x_pred = np.array([[1.5,0.5],[0.5,1.5]])
#
# clf = kNearestNeighbor(4)
# clf.fit(x,y)
# y_pred = clf.predict(x_pred)
#
# print y_pred
