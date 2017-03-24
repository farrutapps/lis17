import csv
import numpy as np
import pickle as pkl
import os

class CsvManager():

  directory = ''

## inititializer

##arguments:
#directory: string of directory where the csv. file shall be stored or opened. 
#           Directory can be either absolute (e.g: C:/User/Documents/......) or relative to working
#           directory (e.g. data)
  def __init__(self, directory):
    self.directory = directory
    

## private function, that merges filename with working directory to a filepath. Should not be used publicly.
  def _filepath(self,filename):
    path = os.path.join(self.directory, filename)
    if not os.path.exists(self.directory):
      os.mkdir(self.directory)

    return path

###save_to_file: 
# saves data to .csv file

## arguments:
# filename: string of filename for storage
# data: np.array containing the data. Ids, input and output need to be merged already
# set_header: string that is the first line in the data file

## return value: None
  def save_to_file(self,filename, data, header):

    path = self._filepath(filename)
    file = open(path, 'wb')
    np.savetxt(file,header, delimiter= ',', fmt ='%s')
    np.savetxt(file,data, delimiter = ',')
    
### restore_from file: 
#Gets data saved in a .csv file. The first line of the datasource is skipped.

## arguments:
#filename: sting specifying the sourcefile. File needs to be of format .csv. 

## return value: 
# data: np.array containing the content of the file.
  def restore_from_file(self, filename):
    path = self._filepath(filename)
    file = open(path, 'rb')
    data = np.loadtxt(file, delimiter=",", skiprows=1)

    return data
