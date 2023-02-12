import numpy as np
from building_blocks import Layer


# loss layer ======================================================================
class CrossEntropyLoss(Layer):
    def __init__(self):
        pass
    
    def forward(self, input, true_labels):
        # input shape is (n_batch, n_classes)
        self.input = input
        self.true_labels = true_labels
        self.batch_size = input.shape[0]
        self.loss = -np.sum(true_labels * np.log(input + 1e-8)) / self.batch_size
        return self.loss
    
    def backward(self):
        output_gradient = -(self.true_labels - self.input) / self.batch_size
        return output_gradient
# end of loss layer ======================================================================