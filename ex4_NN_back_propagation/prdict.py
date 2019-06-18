#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils.sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    # PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    if X.ndim == 1:
        X = np.reshape(X, (-1, X.shape[0]))
    m = np.shape(X)[0]
    num_labels = np.shape(Theta2)[0]

    # You need to return the following variables correctly
    X = np.column_stack((np.ones((m, 1)), X))
    a2 = sigmoid(np.dot(X, Theta1.T))
    a2 = np.column_stack((np.ones((m, 1)), a2))
    a3 = sigmoid(np.dot(a2, Theta2.T))
    p = a3.argmax(axis=1) + 1
    return p
