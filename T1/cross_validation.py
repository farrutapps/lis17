import numpy as np

class CrossValidation():

## inititializer
##arguments:
    #data:    training data which has to be predicted as array of [d+1,n] as follows: y,x_1,...,x_n
    #k:       number of data packages which should be used for the cross validation. k has to be an integer > 1
## public variables:
    #split_data:    #list created by private function _split() containing arrays of separated test data
    #validation_error:    #list containing rms error of individual folds
    def __init__(self,data,k):
        if not k > 1:
            raise ValueError("the number of folds for the cross-validation has to be at least 2")
        self.data = data
        self.k = k
        self.split_data = self._split()
        self.validation_error = [None]*self.k


##private function to split the data, returns list with splits
    def _split(self):
        split_length = int(np.floor(len(self.data[:,0]) / self.k))
        split_data = [None]*self.k
        for i in range(self.k):        
            if i!=(self.k-1): split_data[i] = self.data[i*split_length:(i+1)*split_length,:]
            else: split_data[i] = self.data[i*split_length:,:]
        return split_data
        
        
## public function to start cross-validation-function
## arguments:
    #model:   applied model to achieve fitting
    def start_cross_validation(self, model):
        
        for i in range(self.k):
            
            #combine list elements from split_data[j] where j != i
            for j in range(self.k):
                if i != j:
                    if (i==0 and j==1) or (i!=0 and j==0):
                        predict_data = self.split_data[j]
                    else:
                        predict_data = np.vstack([predict_data,self.split_data[j]])
             
            x_train = predict_data[:,2:]
            y_train = predict_data[:,1]
            x_validate = self.split_data[i][:,2:]
            y_validate = self.split_data[i][:,1]

            model.fit(x_train,y_train)
            self.validation_error[i] = model.validate(x_validate,y_validate)
                 
        return np.average(self.validation_error)







## TESTING






#import linear regression from other branch to test
#############################################################################################
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

###################################################################################


















#fake main program
import os,sys,inspect
from matplotlib.pyplot import waitforbuttonpress
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir) 
import matplotlib.pyplot as plt
import csv_manager as cm

data_manager = cm.CsvManager('C:/Home/Documents/ETH_private/lis17/T1/data')
train_data = data_manager.restore_from_file('train.csv')


cv = CrossValidation(train_data,10)
lm = LinearRegression()
error = cv.start_cross_validation(lm)
print error


'''
error_on_k = [None]*100


for i in range(100):
    cv = CrossValidation(train_data,(i+3))
    lm = LinearRegression()
    error_on_k[i]=cv.start_cross_validation(lm)

plt.plot(np.linspace(10,100,100),error_on_k)
plt.ylabel('average error')
plt.xlabel('number of folds used')
plt.show()
'''