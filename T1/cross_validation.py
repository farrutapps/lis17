import numpy as np
##TODO IMPORT MODELS (LINEAR REGRESSION)


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
        split_length = int(np.round(len(self.data[:,0]) / self.k))
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
                        train_data = self.split_data[j]
                    else:
                        train_data = np.vstack([train_data,self.split_data[j]])
             
            x_train = train_data[:,2:]
            y_train = train_data[:,1]
            x_validate = self.split_data[i][:,2:]
            y_validate = self.split_data[i][:,1]

            model.fit(x_train,y_train)
            self.validation_error[i] = model.validate(x_validate,y_validate)**0.5
                 
        return np.average(self.validation_error)
    