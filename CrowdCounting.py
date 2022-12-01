import numpy as np
from PIL import Image
import os
import random
import scipy.io as sio
import sys
import pickle

from ConvolutionalLayer import Convolutional
from Activations import Tanh
from Losses import mse, mse_prime
from network import train, predict

loadFromMemory = False

projDir = 'C:\\Users\mason\OneDrive\Documents\CNN Crowd Counting'
datasetDir = 'C:\\Users/mason/OneDrive/Documents/UCF-QNRF_ECCV18'

# Loading Methods
def load_resize_format_image(input):
    label = datasetDir + '/Train/' + input
    image = Image.open(label)
    image = image.resize((240, 240))
    numpydata = np.asarray(image) / 255
    return numpydata


def load_annotations_answer(input):
    label = datasetDir + '/Train/' + input
    loadFile = sio.loadmat(label)
    population = len(loadFile['annPoints'])
    answer = np.zeros(15000)
    answer[population] = 1
    return answer

def save_matrix():
    with open('layers.pkl', 'wb') as outp:
        layer1 = network[0]
        pickle.dump(layer1, outp, pickle.HIGHEST_PROTOCOL)
    
        layer2 = network[2]
        pickle.dump(layer2, outp, pickle.HIGHEST_PROTOCOL)

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
    for i in range(1, 10): #1201
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


# load Data
print ('Shuffling Training List')
lst = getRandomizedTrainingDataset()
x_train = []
y_train = []
i = 0
for img in lst:
    i+=1
    x_train.append(load_resize_format_image(img[0]))
    y_train.append(load_annotations_answer(img[1]))
    print('Loading img %d/%d' % (i, len(lst)))

# neural network
network = []

with open('layers.pkl', 'rb') as inp:
    if loadFromMemory:
        print('Loading from memory...')
        layer1 = pickle.load(inp)
        layer2 = pickle.load(inp)
        
        network = [
            layer1,
            Tanh(),
            layer2,
            Tanh()
        ]
    else:
        print('Generating New Matricies')
        network = [
            Convolutional(240 * 240, 175*175, 3),
            Tanh(),
            Convolutional(175*175, 15000, 3),
            Tanh()
        ]

# train
print('Training starts now!')
train(network, mse, mse_prime, x_train, y_train, epochs=100, learning_rate=0.1)
print('Training Complete!')

# For good reason
save_matrix()

# test
print ('Shuffling Training List')
lst = getRandomizedTestingDataset()
x_test = []
y_test = []
i = 0
for img in lst:
    i += 1
    x_train.append(load_resize_format_image(img[0]))
    y_train.append(load_annotations_answer(img[1]))
    print('Loading img %d/%d' % (i, len(lst)))