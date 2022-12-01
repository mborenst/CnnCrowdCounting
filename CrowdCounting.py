import numpy as np
from PIL import Image
import os
import random
import scipy.io as sio
import sys

from DenseLayer import Dense
from Activations import Tanh
from Losses import mse, mse_prime
from network import train, predict


projDir = 'C:\\Users\mason\OneDrive\Documents\CNN Crowd Counting'
datasetDir = 'C:\\Users/mason/OneDrive/Documents/UCF-QNRF_ECCV18'


def load_resize_format_image(input):
    label = datasetDir + '/Train/' + input
    image = Image.open(label)
    image = image.resize((450, 450))
    numpydata = np.asarray(image) / 255
    return numpydata


def load_annotations_answer(input):
    label = datasetDir + '/Train/' + input
    loadFile = sio.loadmat(label)
    population = len(loadFile['annPoints'])
    answer = np.zeros(15000)
    answer[population] = 1
    return answer


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
    for i in range(1, 25): #1201
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
lst = getRandomizedTrainingDataset()
print('List done')
x_train = []
y_train = []
i = 0
for img in lst:
    i+=1
    x_train.append(load_resize_format_image(img[0]))
    y_train.append(load_annotations_answer(img[1]))
    print('Loading img %d/%d' % (i, len(lst)))

"""
1 - Load the Image List for training and testing
2 - Process the images into 450 by 450 numeric matrices
3 - turn correct output into vectors
4 - pass data into network and let the games begin
"""

# neural network
network = [
    Dense(300 * 300, 200*200),
    Tanh(),
    Dense(200*200, 15000),
    Tanh()
]

# train
print('Training starts now!')
train(network, mse, mse_prime, x_train, y_train, epochs=100, learning_rate=0.1)

# test
for x, y in zip(true_x_test_data, true_y_test_data):
    output = predict(network, x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
