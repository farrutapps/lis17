import numpy as np

class cross_validation():

## inititializer

##arguments:
#data:    training data which has to be predicted as array of [d+1,n] as follows: y,x_1,...,x_n
#k:       number of data packages which should be used for the cross validation. k has to be an integer > 1
#model:   applied model to achieve fitting
    def __init__(self,data,k):
        self.data = data
        self.data_length = len(data[:,0])
        self.k = k
        self.model = model
        
        self.split_data = self.split()

#private function to split the data, returns list with splits
    def _split(self):
        split_length = np.floor(self.data_length / self.k)
        split_data = [None]*self.k
        for i in range(self.k):        
            if i!=self.k: split_data[i] = data[i*split_length:(i+1)*split_length,:]
            else: split_data[i] = data[i*split_length:,:]
        return split_data
        
## public function to start cross-validation-function
    def start_cross_validation(self, model):
        for i in range(self.k):
        
            for j in range(self.k):
                if i != j:
                    if (i==0 and j==1) or (i!=0 and j==0):
                        predict_data = self.split_data[j]
                    else:
                        predict_data = np.vstack(predict_data,split_data[j])
                        
            model.predict(predict_data)
            model.validate(self.split_data[i])
        
    return mean_sqare_error
