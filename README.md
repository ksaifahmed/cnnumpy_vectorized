# cnn using numpy but vectorized
Vectorized implementation of the Layers/Building Blocks of a Convolutional Neural Network for faster training and prediction used to Classify Handwritten Bengali Digits.

### Dataset used from a Kaggle Competition: 
[Bengali Handwritten Digit Recognition Competition by Bengal.ai](https://www.kaggle.com/c/numta)

### Layers implemented:
1. Convolution: Four (hyper)parameters:
a. Number of output channels
b. Filter dimension
c. Stride
d. Padding
2. ReLU Activation
3. Max-pooling: Two parameters:
a. Filter dimension
b. Stride
4. Flattening layer: Converts a (series of) convolutional filter maps to a column vector.
5. Fully-connected layer: a dense layer. One parameter: output dimension.
6. Softmax: Converts final layer projections to normalized probabilities.

> Full Specifications of build: [cnn from scratch](https://github.com/ksaifahmed/cnnumpy_vectorized/blob/main/Specifications.pdf)

### Model and Data
* The Architecture used was the standard [LeNet-5](https://en.wikipedia.org/wiki/LeNet)
* The images were preprocessed as follows:
  1. resize to 64 by 64
  2. invert from colors of pixel
  3. transpose to have the shape (batch, channels, image_dimension, image_dimension)

### Training
Training was done using combined images from 'training-a\', 'training-b\' and 'training-c\'. Train-Test was split 80-20. Training was done using mini-batch gradient descent.
* Epochs: 24
* batch size: 32
* Learning Rate: 0.1 (tuned)
* batch size of mini-batch: 32

### Validation
* Accuracy: 0.9525473399458972
* Validation Loss: -7.247408912811393
* Macro f1 score: 0.9523544234682374	, 
* Training Loss: 2.0793054652897394e-05

### Confusion Matrix:
 ![alt text](https://github.com/ksaifahmed/cnnumpy_vectorized/blob/main/results/confusion%20matrix.png?raw=true)

### Metric Graphs: 
 ![alt text](https://github.com/ksaifahmed/cnnumpy_vectorized/blob/main/results/graphs%20of%20metrics.png?raw=true)

### Independent Testing
Testing was done on 'training-d\' as per instructions of project supervisor
* Accuracy: 0.9540704070407041,
* Macro f1 score: 0.9541251631006855

> Full Report on the Results: [Training Results](https://github.com/ksaifahmed/cnnumpy_vectorized/blob/main/results/report.pdf)