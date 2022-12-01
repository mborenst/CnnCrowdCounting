class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    # Receives input, returns output
    def forward(self, input):
        pass
    
    # Updates Trainable parameters if any, and returns the derivative of the error with respect to the input
    def backward(self, output_gradient, learning_rate):
        pass