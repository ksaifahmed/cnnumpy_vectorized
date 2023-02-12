import numpy as np
from building_blocks import Layer


# max pooling layer ======================================================================
class MaxPooling(Layer):
    def __init__(self, filter_width, stride):
        self.filter_width = filter_width
        self.stride = stride
        self.input = None
        self.max_indices = None

    def forward(self, input):
        # vectorized implementation using np.einsum and as_strided
        self.input = input
        n_batch, n_channels, input_width, input_height = input.shape
        output_width = (input_width - self.filter_width) // self.stride + 1
        output_height = (input_height - self.filter_width) // self.stride + 1
        output = np.zeros((n_batch, n_channels, output_width, output_height))

        # create a view of the input array with the shape of the output array
        # and the strides of the filter
        input_view = np.lib.stride_tricks.as_strided(input, shape=(n_batch, n_channels, output_width, output_height, self.filter_width, self.filter_width), strides=(input.strides[0], input.strides[1], self.stride * input.strides[2], self.stride * input.strides[3], input.strides[2], input.strides[3]))
        # compute the max along the last two dimensions
        output = input_view.max(axis=4).max(axis=4)
        # compute the indices of the max values
        self.max_indices = input_view.argmax(axis=4).argmax(axis=4)
        return output
    
    def backward(self, output_gradient, learning_rate):
        # vectorized implementation using np.einsum and as_strided
        # backward can only be vectorized when stride == filter_width
        assert self.stride == self.filter_width
        n_batch, n_channels, output_width, output_height = output_gradient.shape
        input_gradient = np.zeros_like(self.input)
        # create a view of the input gradient array with the shape of the output gradient array
        # and the strides of the filter
        input_gradient_view = np.lib.stride_tricks.as_strided(input_gradient, shape=(n_batch, n_channels, output_width, output_height, self.filter_width, self.filter_width), strides=(input_gradient.strides[0], input_gradient.strides[1], self.stride * input_gradient.strides[2], self.stride * input_gradient.strides[3], input_gradient.strides[2], input_gradient.strides[3]))
        # set the max values to the output gradient
        input_gradient_view[np.arange(n_batch)[:, None, None, None], np.arange(n_channels)[None, :, None, None], np.arange(output_width)[None, None, :, None], np.arange(output_height)[None, None, None, :], self.max_indices // self.filter_width, self.max_indices % self.filter_width] = output_gradient
        return input_gradient
# end of max pooling layer ======================================================================