import numpy as np
import pickle
import cv2
import os
from tqdm import tqdm

from convolution import Conv2D
from activations import ReLU, Softmax
from pooling import MaxPooling
from flattening import Flatten
from linear_layer import Dense
from model import ConvolutionalNetwork


def load_images(folder_path, c_dim=64):
    filenames = [name for name in os.listdir(folder_path)]
    image_paths = [folder_path + name for name in filenames]

    X = [
        cv2.resize(
            255 - cv2.imread(path, cv2.IMREAD_COLOR), (c_dim, c_dim)
        ) for path in tqdm(image_paths, desc='Loading images:')
    ]    
    X = np.array(X) 
    
    # X.shape is (batch, c_dim, c_dim, channels)
    # convert to (batch, channels, c_dim, c_dim)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X / 255.0  # normalize
    return X, filenames



if __name__ == '__main__':
    # the model ===========================================================================
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


    with open('1705110_model.pickle', 'rb') as f:
        for layer in convolutional_network.layers:
            if isinstance(layer, Conv2D):
                layer.weights = pickle.load(f)
                layer.biases = pickle.load(f)
            elif isinstance(layer, Dense):
                layer.weights = pickle.load(f)
                layer.bias = pickle.load(f)

    # evaluate model
    # from data_handler import load_data
    # X_test, y_test = load_data('training-d.csv', 'training-d/', num_classes=10, c_dim=64)
    # convolutional_network.evaluate(X_test, y_test, tq=True)

    # enter datapath to image folder
    datapath = input("Enter path to folder: ")
    if(datapath[-1] != '/'):
        datapath += '/'

    X_test, filenames = load_images(datapath, 64)
    y_predictions = convolutional_network.predict(X_test, tq=True)
    with open('1705110_prediction.csv', 'w') as f:
        f.write('FileName,Digit\n')
        for i in range(len(filenames)):
            f.write('{},{}\n'.format(filenames[i], y_predictions[i]))
    