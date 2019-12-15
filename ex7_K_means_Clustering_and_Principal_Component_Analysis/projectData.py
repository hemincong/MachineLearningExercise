#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# PROJECTDATA Computes the reduced data representation when projecting only
# on to the top k eigenvectors
#   Z = projectData(X, U, K) computes the projection of
#   the normalized inputs X into the reduced dimensional space spanned by
#   the first K columns of U. It returns the projected examples in Z.
#

# You need to return the following variables correctly.
def projectData(X, U, K):
    # You need to return the following variables correctly.
    Z = np.zeros((np.shape(X)[0], K))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the projection of the data using only the top K
    # eigenvectors in U(first K columns).
    # For the i - th example X(i,:), the projection on to the k - th
    # eigenvector is given as follows:
    # x = X(i,:)';
    # projection_k = x ' * U(:, k);
    #

    u_reduce = U[:, :K]
    projection_k = X.dot(u_reduce)
    return projection_k
