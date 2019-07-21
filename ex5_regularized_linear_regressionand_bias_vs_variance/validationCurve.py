#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def validationCurve(X, y, Xval, yval):
    # VALIDATIONCURVE Generate the train and validation errors needed to
    # plot a validation curve that we can use to select lambda
    #       [lambda_vec, error_train, error_val] = ...
    #             VALIDATIONCURVE(X, y, Xval, yval) returns the train
    #             and validation errors( in error_train, error_val)
    #             for different values of lambda.You are given the training set (X,
    #             y) and validation set (Xval, yval).

    #
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = []
    error_val = []

    from ex5_regularized_linear_regressionand_bias_vs_variance.trainLinearReg import trainLinearReg
    from ex5_regularized_linear_regressionand_bias_vs_variance.linearRegCostFunction import linearRegCostFunction
    for l in lambda_vec:
        _, theta = trainLinearReg(X, y, l)
        error_train.append(linearRegCostFunction(X, y, theta, 0)[0])
        error_val.append(linearRegCostFunction(Xval, yval, theta, 0)[0])

    return lambda_vec, error_train, error_val
