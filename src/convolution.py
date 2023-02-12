import numpy as np
from tqdm import tqdm
from building_blocks import Layer


# convolutional layer ======================================================================
class Conv2D(Layer):
    def __init__(self, n_output_channels, filter_wh, stride=1, padding=0):
        self.n_output_channels = n_output_channels # number of filters
        self.filter_wh = filter_wh
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = None
        self.input = None
        self.cache = None

    
    def init_weights_biases(self, input_shape):
        # Xavier initialization
        num_channels = input_shape[1]
        self.weights = np.random.randn(self.n_output_channels, num_channels, self.filter_wh, self.filter_wh) * \
                        np.sqrt(2 / (self.filter_wh * self.filter_wh * num_channels))

        self.biases = np.zeros(self.n_output_channels)

    
    def forward(self, input):
        n, c, h, w = input.shape
        output_height = (h + 2 * self.padding - self.filter_wh ) // self.stride + 1
        output_width = (w + 2 * self.padding - self.filter_wh ) // self.stride + 1
        output = np.zeros((n, self.n_output_channels, output_height, output_width))

        if self.weights is None:
            self.init_weights_biases(input.shape)

        padded_input = np.pad(input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        batch_stride, channel_stride, height_stride, width_stride = padded_input.strides
        strided_windows = np.lib.stride_tricks.as_strided(
            padded_input,
            shape = (n, c, output_height, output_width, self.filter_wh, self.filter_wh),
            strides = (batch_stride, channel_stride, height_stride * self.stride, width_stride * self.stride, height_stride, width_stride)
        )

        output = np.einsum('bihwkl,oikl->bohw', strided_windows, self.weights)
        output += self.biases[None, :, None, None]
        self.cache = input, strided_windows
        return output

    def backward(self, output_gradient, learning_rate):
        input, strided_windows = self.cache
        padding = self.filter_wh - 1 if self.padding == 0 else self.padding
        dilate = self.stride - 1
        temp_out_grad = output_gradient.copy()
        
        if dilate > 0:
            temp_out_grad = np.insert(temp_out_grad, range(1, output_gradient.shape[2]), 0, axis=2)
            temp_out_grad = np.insert(temp_out_grad, range(1, output_gradient.shape[3]), 0, axis=3)
        
        if padding > 0:
            temp_out_grad = np.pad(temp_out_grad, ((0,0), (0,0), (padding, padding), (padding, padding)), 'constant')

        _, _, out_h, out_w = input.shape
        out_b, out_c, _, _ = output_gradient.shape
        batch_stride, channel_stride, height_stride, width_stride = temp_out_grad.strides

        dout_windows = np.lib.stride_tricks.as_strided(temp_out_grad, 
            shape = (out_b, out_c, out_h, out_w, self.filter_wh, self.filter_wh),
            strides = (batch_stride, channel_stride, height_stride * 1, width_stride * 1, height_stride, width_stride)
        )
        rotate_weights = np.rot90(self.weights, 2, (2, 3))

        db = np.sum(output_gradient, axis=(0, 2, 3))
        dw = np.einsum('bihwkl, bohw -> oikl', strided_windows, output_gradient)
        dx = np.einsum('bohwkl, oikl -> bihw', dout_windows, rotate_weights)

        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        
        return dx
# end of convolutional layer ======================================================================