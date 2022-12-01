from Layer import Layer
import numpy as np

class Dense(Layer):
    # Random Startup
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    # Y = W * X + B
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    #e/w = e/y*x^t
    #e/b = e/y
    #e/x = w^t*e/y
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)
    
    