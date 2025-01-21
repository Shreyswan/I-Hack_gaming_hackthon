#######################################################
# Owners: Aditi Pawar, Himanish Goel, Shreyas Sawant
# File name: data_handler.py
# Purpose: Read data and and clean the tweets.
#######################################################

import os
import cv2
import itertools
import numpy as np
from config import *
from sklearn.model_selection import train_test_split

def image_data_reader():
    image_list = []
    labels = []
    for folder in os.listdir(DATA_PATH_SIAMESE):
        filepath = os.path.join(DATA_PATH_SIAMESE, folder)
        if os.path.isdir(filepath):
            for image in os.listdir(filepath):
                try:
                    img = cv2.imread(os.path.join(filepath, image))
                    img = img.astype("float32") / 255.0
                    image_list.append(img)
                    labels.append(folder)

                except:
                    print(image)
    
    image_list =  np.array(image_list)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(image_list, labels, test_size = 0.15, random_state = 4)

    return X_train, X_test, y_train, y_test

def make_paired_dataset(X, y):
    X_pairs, y_pairs = [], []
    tuples = [(x1, y1) for x1, y1 in zip(X, y)]

    for t in itertools.product(tuples, tuples):
        pair_A, pair_B = t
        img_A, label_A = t[0]
        img_B, label_B = t[1]
        new_label = int(label_A == label_B)
        X_pairs.append([img_A, img_B])
        y_pairs.append(new_label)

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)
    return X_pairs, y_pairs

# if __name__ == '__main__':
#     image_data_reader()