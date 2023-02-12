import numpy as np
from building_blocks import Layer


# dense layer ======================================================================
class Dense(Layer):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.weights = None
        self.bias = None
        self.input = None

    def forward(self, input):
        self.input = input
        if self.weights is None:
            self.weights = np.random.randn(input.shape[1], self.output_dim) * np.sqrt(2 / input.shape[1])
            self.bias = np.zeros(self.output_dim)
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * np.dot(self.input.T, output_gradient)
        self.bias -= learning_rate * np.sum(output_gradient, axis=0)
        return input_gradient
# end of dense layer ======================================================================