import os
import cv2
import numpy as np

def load_data():
    path = "./DataSets/number"
    labels = os.listdir()
    Y = []
    X = []
    for dirpath, dirnames, filenames in os.walk(path):
        print(dirpath)
        for filename in filenames:
            name = dirpath.split("/")
            val = cv2.imread(dirpath+"/"+filename, 0)
            val = cv2.resize(val, (28,28))
            X.append(val)
            val = cv2.GaussianBlur(val, (9,9), 0)
            val = cv2.adaptiveThreshold(val, 255, 1, 1, 19, 5)
            X.append(val)
            Y.append(int(name[-1]))
            Y.append(int(name[-1]))
    Y = np.asarray(Y)
    X = np.asarray(X)
    print(len(X), len(Y))
    return X, Y

