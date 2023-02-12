import numpy as np
from building_blocks import Layer


# Activations ======================================================================
class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)   
    

class Softmax(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, output_gradient, learning_rate):
        return output_gradient
# end of activations ======================================================================