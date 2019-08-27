#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import scipy.io


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
    def test_load_and_visualzing_data(self):
        # Extract Features

        with open('resource/emailSample1.txt', 'r') as emailSample:
            file_contents = emailSample.read()
            from ex6_SVM.processEmail import processEmail
            word_indices = processEmail(file_contents, self.vocabList)
            from ex6_SVM.emailFeatures import emailFeatures
            features = emailFeatures(word_indices)

            # Print Stats
            print('Word Indices:')
            print('{word_indices}'.format(word_indices=word_indices))
            self.assertGreater(len(features), 0)
            self.assertGreater(len(word_indices), 0)


