import numpy as np
from building_blocks import Layer


# flatten layer ======================================================================
class Flatten(Layer):
    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        n_batch, n_channels, height, width = input.shape
        return input.reshape(n_batch, n_channels * height * width)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)
# end of flatten layer ======================================================================