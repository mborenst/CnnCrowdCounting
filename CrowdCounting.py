import numpy as np
from PIL import Image
import os
import random
import scipy.io as sio
import sys
import pickle

from ConvolutionalLayer import Convolutional
from DenseLayer import Dense
from ReshapeLayer import Reshape
from Activations import Sigmoid
from Losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict

loadFromMemory = False
imageSize = 240

projDir = 'C:\\Users\mason\OneDrive\Documents\CNN Crowd Counting'
datasetDir = 'C:\\Users/mason/OneDrive/Documents/UCF-QNRF_ECCV18'

# Loading Methods
def load_resize_format_image(input):
    label = datasetDir + '/Train/' + input
    image = Image.open(label)
    image = image.resize((imageSize, imageSize))
    numpydata = np.asarray(image)
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
    
        layer2 = network[3]
        pickle.dump(layer2, outp, pickle.HIGHEST_PROTOCOL)
        
        layer3 = network[5]
        pickle.dump(layer3, outp, pickle.HIGHEST_PROTOCOL)

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
    for i in range(1, 50): #1201
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
    i = random.randint(1, 16) # 334
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
x = []
y = []
i = 0
for img in lst:
    thing = np.array(load_resize_format_image(img[0]))
    x.append(thing.reshape(3, imageSize, imageSize))
    y.append(np.array(load_annotations_answer(img[1])))
    i+=1
    print('Loading img %d/%d' % (i, len(lst)))


print(x[0].shape)
x_train = np.stack(x, axis = 0)
print(x_train.shape)
#x_train = x_train.reshape(len(lst), 1, imageSize, imageSize)
y_train = np.stack(y, axis = 0)
print(x_train.shape)
#y_train = np.reshape(len(lst), 15000, 1)

# neural network
network = []

with open('layers.pkl', 'rb') as inp:
    if loadFromMemory:
        print('Loading from memory...')
        layer1 = pickle.load(inp)
        layer2 = pickle.load(inp)
        layer3 = pickle.load(inp)
        
        network = [
            layer1,
            Sigmoid(),
            Reshape((5, 26, 26), (5*26*26, 1)),
            layer2,
            Sigmoid(),
            layer3,
            Sigmoid()
        ]
    else:
        print('Generating New Matricies')
        network = [
            Convolutional((1, imageSize, imageSize), 3, 5),
            Sigmoid(),
            Reshape((5, imageSize-2, imageSize-2), (5*(imageSize-2)*(imageSize-2), 1)),
            Dense(5*(imageSize-2)*(imageSize-2), 175*175),
            Sigmoid(),
            Dense(175*175, 15000),
            Sigmoid()
        ]

# train
print('Training starts now!')

train(network, binary_cross_entropy, binary_cross_entropy_prime, x_train, y_train, epochs=50, learning_rate=0.1)

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