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
from Activations import Sigmoid, Tanh
from Losses import binary_cross_entropy, binary_cross_entropy_prime, mse, mse_prime
from network import train, predict

loadFromMemory = False
imageSize = 240

datasetDir = 'UCF-QNRF_ECCV18'

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
    answer = [population / 12865]
    return answer


def save_errors():
    with open('errors_tanh_bse.pkl', 'wb') as outp:
        pickle.dump(e1, outp, pickle.HIGHEST_PROTOCOL)

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
    for i in range(1, 301):  # 1201
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
    # random.shuffle(lst)
    return lst


def getRandomTestImage():
    i = random.randint(1, 16)  # 334
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
    for i in range(1, 34):  # 334
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
print('Shuffling Training List')
lst = getRandomizedTrainingDataset()
x = []
y = []
i = 0
for img in lst:
    thing = np.array(load_resize_format_image(img[0]))
    print('Loading img %d/%d (%s)' % (i, len(lst), img[0]))
    if (thing.size == 172800):
        x.append(thing.reshape(3, imageSize, imageSize))
        y.append(np.array(load_annotations_answer(img[1])))
    else:
        print("Error: size="+str(thing.size))
    i += 1

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
            Reshape((5, imageSize-2, imageSize-2),
                    (5*(imageSize-2)*(imageSize-2), 1)),
            layer2,
            Sigmoid(),
            layer3,
            Sigmoid()
        ]
    else:
        print('Generating New Matricies')
        network = [
            Convolutional((1, imageSize, imageSize), 3, 5),
            Tanh(),
            Reshape((5, imageSize-2, imageSize-2),
                    (5*(imageSize-2)*(imageSize-2), 1)),
            Dense(5*(imageSize-2)*(imageSize-2), 100),
            Tanh(),
            Dense(100, 1),
            Tanh()
        ]

# Training time
print('Training MSE Signmoid now!')
e1 = train(network, binary_cross_entropy, binary_cross_entropy_prime, x, y, epochs=100, learning_rate=0.1)
print('Training Complete!')

# For good reason
save_errors()

# test
print('Shuffling Training List')
lst = getRandomizedTestingDataset()
x_test = []
y_test = []
i = 0
for img in lst:
    i += 1
    x_test.append(load_resize_format_image(img[0]))
    y_test.append(load_annotations_answer(img[1]))
    print('Loading img %d/%d (%s)' % (i, len(lst), img[0]))

for X, Y in zip(x_test, y_test):
    pred = predict(network, X)
    err = binary_cross_entropy(Y[0], pred)
    print('Predicted=%d  Actual=%d  Error=%f' %
          (pred * 12865, Y[0] * 12865, err))
