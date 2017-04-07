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
        if not (k > 1 and k<=data.shape[0]):
            raise ValueError("the number of folds for the cross-validation has to be between 2 and the number of samples")
        self.data = data
        self.k = k
        self.split_data = self._split()
        self.validation_error = np.empty(self.k)

##private function to split the data, returns list with splits
    def _split(self):
        split_length = int(np.round(self.data.shape[0]) / self.k)
        split_data = np.empty(shape=(self.k,),dtype = object)
        for i in range(self.k):
            if i!=(self.k-1):
                split_data[i] = self.data[i*split_length:(i+1)*split_length,:]
            else:
                split_data[i] = self.data[i*split_length:,:]
        return split_data

## public function to start cross-validation-function
## arguments:
    #model:   applied model to achieve fitting
## returns: average validation error
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
            x_validate = np.array(self.split_data[i][:,2:])
            y_validate = np.array(self.split_data[i][:,1])

            model.fit(x_train,y_train)
            self.validation_error[i] = model.validate(x_validate,y_validate)**0.5

        return np.average(self.validation_error)


## This function randomly chooses one of the split_data sets. It trains the model on all but the chosen one, which is used to validate the data.
## It does this just once. For high k this is much faster than start_cross_validation but also has a worse estimation of the validation_error.
    def start_single_validation(self, model):
        i = int(np.random.uniform(0,self.k))

        train_data = np.vstack(self.split_data[:])
        train_data = np.delete(train_data, i,0)

        x_train = train_data[:,2:]
        y_train = train_data[:,1]
        x_validate = np.array(self.split_data[i][:,2:])
        y_validate = np.array(self.split_data[i][:,1])

        model.fit(x_train,y_train)
        self.validation_error = model.validate(x_validate,y_validate)**0.5

        return self.validation_error
