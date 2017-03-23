# load data
import CsvManager as csv_man
import numpy as np

def predict_mean(data):
  prediction = np.empty((data.shape[0]+1,2))
  prediction[0,0] = 1
  prediction[0,1] = 2

  for i in range(data.shape[0]):
  	
  	id_ = data[i,0]
  	y = np.mean(data[i,1:])

  	prediction[i+1,0] = id_ 
  	prediction[i+1,1] = y

  return prediction

data_manager = csv_man.CsvManager('data')
train_data = data_manager.restore_from_file('train.csv')


### check if data can really be modeled as y = mean(x)
# 
#success = True
# i = 0
# while success:
  
#   mean = np.mean(data[i,2:])
#   y = train_data[i,1]

#   if y != mean:
#   	success = False
#   	print('mean {} y={}'.format(mean,y))

#   else:
#   	print('yes')
    

test_data = data_manager.restore_from_file('test.csv')

prediction = predict_mean(test_data)

data_manager.save_to_file('result.csv',prediction, True)
