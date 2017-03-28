import numpy as np
import csv_manager as cm
import sys

# supress .pyc files
sys.dont_write_bytecode = True

class Feature:

	### initializer:
	## arguments:
	# x_column_index: array listing the indices of the data points that shall be multiplied
	# method name: name of the method that shall be applied to the data on evaluation. Must be listed in the method register_methods.
	def __init__(self, x_column_index, method_name):

		self._register_methods()
		self.x_column_index = x_column_index.reshape(max(x_column_index.shape))
		self.method_name = method_name

	## Private method. do not use from outside the class.
	# Used to initialize the list of feature models.
	def _register_methods(self):

		# NOTE: the keys of the dictionaries are strings. If necessary to iterate through the entries of the dicitonary, use function dict.keys() to obtain a list.
		self.methods = {}

		self.methods['multiply']= self.multiply
		self.methods['exp'] = self.exp
		self.methods['log'] = self.log

	### evaluate:
	## arguments:
	# x: np.array with one dimension = 1. The data it contains will be evaluated corresponding to the indexes and method_name specified when initilializing the class.
	def evaluate(self, x):
		return self.methods[self.method_name](x, self.x_column_index)

	### multiply:
	# multiplies the elements of data specified by the indexes in x_column_index.
	#
	## arguments:
	# x: np.array with one dimension = 1.
	# x_column_index: array listing the ids of the data points that shall be multiplied
	#
	## return value:
	# result of the multiplication
	def multiply(self,x, x_column_index):
		result = 1

		for i in x_column_index:
			result *= x[i]

		return result

	### exp:
	# returns exp(xi + xj + ...) of the specified data.
	# arguments see method 'multiply'
	def exp(self, x, x_column_index):
		x_sum = 0

		for i in x_column_index:
			x_sum += x[i]

		return np.exp(x_sum)

	### log:
	# returns log(xj * xi * ...) of the specified data
	def log(self, x, x_column_index):
		return np.log(self.multiply(x,x_column_index))



#TODO: Discuss how to handle functions exp and ln when x has more than one element. Excepiton? Define otherwise?


### Testing Sebastian, Sam

# man = cm.CsvManager('data')
# data = man.restore_from_file('test.csv')

# indexes = np.array([0,1]).reshape(1,2)
# feature = Feature(indexes,'multiply')

# for line in data[0:5,1:15]:
# 	print feature.evaluate(line)
