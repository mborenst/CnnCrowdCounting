# CNN Main File
import os
import scipy.io as sio
import numpy as np
import h5py

projDir = 'C:\\Users\\mason\\OneDrive\\Documents\\CNN Crowd Counting'
datasetDir = 'C:\\Users\\mason\\OneDrive\\Documents\\UCF-QNRF_ECCV18'

print(os.listdir(projDir+'\\')) # returns list

test = sio.loadmat(datasetDir+'\\img_0001_ann.mat')
# print(test)
data = test['annPoints']
print(type(data))
#print(data)
print(len(data))