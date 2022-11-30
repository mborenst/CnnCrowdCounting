# CNN Main File
import os
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 
import random

projDir = 'C:\\Users\mason\OneDrive\Documents\CNN Crowd Counting'
datasetDir = 'C:\\Users/mason/OneDrive/Documents/UCF-QNRF_ECCV18'

#print(os.listdir(projDir+'/')) # returns list
#print(os.listdir(datasetDir+'/Train')) # returns list

def getRandomizedDataset():
    lst = []
    for i in range(1,1201):
        label = 'img_'
        if (i < 10):
            label = label + "000" + str(i)
        elif (i < 100):
            label = label + "00" + str(i)
        elif (i < 1000):
            label = label + "0" + str(i)
        else:
            label = label + str(i)
        image = label + ".jpg"
        annotation = label + "_ann.mat"
        lst.append((image, annotation))
    random.shuffle(lst)
    return lst     



def compareResult(output, img, ann):
    calculated = getNumberOfPeople(output)
    loadFile = sio.loadmat(datasetDir+'/Train/'+ann)
    actual = len(loadFile['annPoints'])
    return actual

def getNumberOfPeople(output):
    lst = getAllPeaks(output)
    population = len(lst)
    return population

def getAllPeaks(lst):
    peaks, _ = find_peaks(lst, prominence=.60)
    return peaks

lst = getRandomizedDataset()
for i in lst:
    print(compareResult([], i[0], i[1]))