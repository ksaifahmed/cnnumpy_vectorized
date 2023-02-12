import numpy as np
import pickle

from data_handler import load_data
from convolution import Conv2D
from activations import ReLU, Softmax
from pooling import MaxPooling
from flattening import Flatten
from linear_layer import Dense
from model import ConvolutionalNetwork



if __name__ == '__main__':
    # the data ===========================================================================
    csv_paths = ['/kaggle/input/numta/training-a.csv', '/kaggle/input/numta/training-b.csv', '/kaggle/input/numta/training-c.csv']
    img_path = ['/kaggle/input/numta/training-a/', '/kaggle/input/numta/training-b/', '/kaggle/input/numta/training-c/']

    # load data from csv files
    X_a, y_a = load_data(csv_paths[0], img_path[0], num_classes=10, c_dim=64)
    X_b, y_b = load_data(csv_paths[1], img_path[1], num_classes=10, c_dim=64)
    X_c, y_c = load_data(csv_paths[2], img_path[2], num_classes=10, c_dim=64)

    # concatenate data
    X_ = np.concatenate((X_a, X_b, X_c), axis=0)
    y_ = np.concatenate((y_a, y_b, y_c), axis=0)

    # split data into train and test 80/20
    X_train, X_test = X_[:int(0.8*len(X_))], X_[int(0.8*len(X_)):]
    y_train, y_test = y_[:int(0.8*len(y_))], y_[int(0.8*len(y_)):]

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # end of data ===========================================================================



    # the model: LeNet-5 ===========================================================================
    convolutional_network = ConvolutionalNetwork()
    convolutional_network.build_netword([
        Conv2D(6, 5, 1, 0),
        ReLU(),
        MaxPooling(2, 2),
        Conv2D(16, 5, 1, 0),
        ReLU(),
        MaxPooling(2, 2),    
        Flatten(),
        Dense(120),
        ReLU(),
        Dense(84),
        ReLU(),
        Dense(10),
        ReLU(),
        Softmax()
    ])

        
    # train the network
    convolutional_network.train(X_train, y_train, X_test, y_test, 0.1, 25, 32)
    # ======================================================================================


    # write Conv2D and Dense weights to pickle file
    with open('1705110_model.pickle', 'wb') as f:
        for layer in convolutional_network.layers:
            if isinstance(layer, Conv2D):
                pickle.dump(layer.weights, f)
                pickle.dump(layer.biases, f)
            elif isinstance(layer, Dense):
                pickle.dump(layer.weights, f)
                pickle.dump(layer.bias, f)