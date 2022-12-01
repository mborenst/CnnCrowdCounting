import os
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 
import random
import time

projDir = 'C:\\Users\mason\OneDrive\Documents\CNN Crowd Counting'
datasetDir = 'C:\\Users/mason/OneDrive/Documents/UCF-QNRF_ECCV18'

#print(os.listdir(projDir+'/')) # returns list
#print(os.listdir(datasetDir+'/Train')) # returns list

def getRandomTraining():
    i = random.randint(1, 1201)
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
    return (image, annotation)

def getRandomizedTrainingDataset():
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

def getRandomTestImage():
    i = random.randint(1, 334)
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
    return (image, annotation)

def getRandomizedTestingDataset():
    lst = []
    for i in range(1,334):
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

lst = getRandomizedTestingDataset()
largest = -1
for i in lst:
    r = compareResult([], i[0], i[1])
    if (r > largest):
        largest = r
        print(largest)

# 12865 Training
# 11434 Testing

# Output Vector: 13000 Vector of crowd size