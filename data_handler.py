import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


def load_data(label_path, image_path, num_classes, c_dim=64, gray_sc=False):
    df = pd.read_csv(label_path)
    y = df['digit'].values
    y = np.array(y) # y.shape (batch, 1)
    y = np.eye(num_classes)[y] # one hot encoding

    image_names = df['filename'].values
    image_paths = [image_path + name for name in image_names]
    
    # resize all images to c_dim x c_dim
    mode = cv2.IMREAD_GRAYSCALE if gray_sc else cv2.IMREAD_COLOR        
    X = [
        cv2.resize(
            255 - cv2.imread(path, mode), (c_dim, c_dim)
        ) for path in tqdm(image_paths, desc='Loading images:')
    ]    
    X = np.array(X) 

    # add channel dimension if grayscale
    if gray_sc:
        X = np.expand_dims(X, axis=-1) # add channel dimension
    
    # X.shape is (batch, c_dim, c_dim, channels)
    # convert to (batch, channels, c_dim, c_dim)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X / 255.0  

    return X, y