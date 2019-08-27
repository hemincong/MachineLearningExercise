#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


# Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
class test_ex6_spam(unittest.TestCase):

    @classmethod
    def setUp(cls):
        from ex6_SVM.getVocabList import getVocabList
        cls.vocabList = getVocabList("resource/vocab.txt")

    # ==================== Part 1: Email Preprocessing ====================
    # To use an SVM to classify emails into Spam v.s.Non - Spam, you first need
    # to convert each email into a vector of features.In this part, you will
    # implement the preprocessing steps for each email.You should
    # complete the code in processEmail.m to produce a word indices vector
    # for a given email.
    def test_email_preprocessing(self):
        # Extract Features

        with open('resource/emailSample1.txt', 'r') as emailSample:
            file_contents = emailSample.read()
            from ex6_SVM.processEmail import processEmail
            word_indices = processEmail(file_contents, self.vocabList)

            # Print Stats
            print('Word Indices:')
            print('{word_indices}'.format(word_indices=word_indices))
            self.assertGreater(len(word_indices), 0)

    # ==================== Part 2 : Feature Extraction====================
    # Now, you will convert each email into a vector of features in R ^ n.
    # You should complete the code in emailFeatures.m to produce a feature
    # vector for a given email.
    def test_feature_extraction(self):
        # Extract Features

        with open('resource/emailSample1.txt', 'r') as emailSample:
            file_contents = emailSample.read()
            from ex6_SVM.processEmail import processEmail
            word_indices = processEmail(file_contents, self.vocabList)
            from ex6_SVM.emailFeatures import emailFeatures
            features = emailFeatures(word_indices)

            self.assertGreater(len(features), 0)

            # Print Stats
            print('Length of feature vector: {num_of_features}'.format(num_of_features=len(features)))
            print('Number of non-zero entries: {sum_of_feature}'.format(sum_of_feature=sum(features > 0)[0]))

    # =========== Part 3: Train Linear SVM for Spam Classification ========
    #  In this section, you will train a linear classifier to determine if an
    #  email is Spam or Not-Spam.

    # Load the Spam Email dataset
    # You will have X, y in your environment
    def test_spam_train(self):
        import scipy.io
        mat = scipy.io.loadmat('resource/spamTrain.mat')
        X = mat["X"]
        y = mat["y"]

        y = y.flatten()

        print('Training Linear SVM (Spam Classification)')
        print('(this may take 1 to 2 minutes) ...')

        C = 0.1
        from ex6_SVM.svmTrain import svmTrain
        model = svmTrain(X, y, C, "linear")

        p = model.predict(X)
        import numpy as np
        ret = np.mean((p == y)) * 100
        print("Training Accuracy: {accuracy}".format(accuracy=ret))
