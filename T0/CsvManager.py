import csv
import numpy as np
import pickle as pkl
import os

class CsvManager():

  directory = ''

  def __init__(self, directory):
    self.directory = directory
    

  def filepath(self,filename):
    path = os.path.join(self.directory, filename)
    if not os.path.exists(self.directory):
      os.mkdir(self.directory)

    return path

  def save_to_file(self,filename, data, set_header = False):
    

    path = self.filepath(filename)
    file = open(path, 'wb')

    if set_header:
      header =np.array(['Id,y'])
      np.savetxt(file,header, delimiter= ',', fmt ='%s')
    
    np.savetxt(file,data, delimiter = ',')
    

  def restore_from_file(self, filename):
    path = self.filepath(filename)
    file = open(path, 'rb')
    data = np.loadtxt(file, delimiter=",", skiprows=1)

    return data
