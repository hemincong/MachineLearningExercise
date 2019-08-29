#!/usr/bin/env python
# -*- coding: utf-8 -*-


def dataset3Params(X, y, Xval, yval):
    # EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
    # where you select the optimal (C, sigma) learning parameters to use for SVM
    # with RBF kernel
    #   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
    #   sigma. You should complete this function to return the optimal C and
    #   sigma based on a cross-validation set.
    #

    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    # learning parameters found using the cross validation set.
    # You can use svmPredict to predict the labels on the cross
    # validation set.For example,
    # predictions = svmPredict(model, Xval);
    # will return the predictions on the cross validation set.
    #
    # Note: You can compute the prediction error using
    # mean(double(predictions ~ = yval))
    #
    C_vec = [0.01, 0.03, 0.1, 0.3, 1.3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1.3, 10, 30]

    import numpy as np
    ret = []

    from ex6_SVM.svmTrain import svmTrain
    from ex6_SVM.gaussianKernelGramMatrix import gaussianKernelGramMatrix
    for c in C_vec:
        for s in sigma_vec:
            model = svmTrain(X, y, c, "gaussian", sigma=s)
            p = model.predict(gaussianKernelGramMatrix(Xval, X))
            error = np.mean((p != yval)) * 100
            ret.append((c, s, error))

    min_e = 100
    min_c = 100
    min_s = 100

    for r in ret:
        C, sigma, e = r
        print("sigma={sigma}, C={c}, error={error}".format(sigma=sigma, c=C, error=e))
        if e < min_e:
            min_c = C
            min_s = sigma
            min_e = e

    print("min sigma={sigma}, min C={c}, min error={error}".format(sigma=min_s, c=min_s, error=min_e))
    return min_c, min_s
