#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

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

        mat = scipy.io.loadmat('resource/spamTrain.mat')
        cls.X = mat["X"]
        cls.y = mat["y"]

        cls.y = cls.y.flatten()

        print('Training Linear SVM (Spam Classification)')
        print('(this may take 1 to 2 minutes) ...')

        C = 0.1
        from ex6_SVM.svmTrain import svmTrain
        cls.model = svmTrain(cls.X, cls.y, C, "linear")

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
        p = self.model.predict(self.X)
        import numpy as np
        ret = np.mean((p == self.y)) * 100
        print("Training Accuracy: {accuracy}".format(accuracy=ret))
        self.assertAlmostEqual(ret, 99.85, delta=0.1)

        # =================== Part 4: Test Spam Classification ================
        #  After training the classifier, we can evaluate it on a test set. We have
        #  included a test set in spamTest.mat

        # Load the test dataset
        # You will have Xtest, ytest in your environment
        mat = scipy.io.loadmat('resource/spamTest.mat')
        Xtest = mat["Xtest"]
        ytest = mat["ytest"]

        ytest = ytest.flatten()

        print('Evaluating the trained Linear SVM on a test set ...')

        p = self.model.predict(Xtest)
        ret = np.mean((p == ytest)) * 100
        print('Test Accuracy: {accuracy}'.format(accuracy=ret))
        self.assertAlmostEqual(ret, 98.9, delta=0.1)

    # ================= Part 5: Top Predictors of Spam ====================
    #  Since the model we are training is a linear SVM, we can inspect the
    #  weights learned by the model to understand better how it is determining
    #  whether an email is spam or not. The following code finds the words with
    #  the highest weights in the classifier. Informally, the classifier
    #  'thinks' that these words are the most likely indicators of spam.
    #
    def test_predictors_of_spam(self):
        # Sort the weights and obtain the vocabulary list
        w = self.model.coef_[0]

        # from http://stackoverflow.com/a/16486305/583834
        # reverse sorting by index
        indices = w.argsort()[::-1][:15]
        vocabList = sorted(self.vocabList.keys())

        print('Top predictors of spam: ')
        for idx in indices:
            print(' {:s} ({:f}) '.format(vocabList[idx], float(w[idx])))

    # =================== Part 6: Try Your Own Emails =====================
    #  Now that you've trained the spam classifier, you can use it on your own
    #  emails! In the starter code, we have included spamSample1.txt,
    #  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
    #  The following code reads in one of these emails and then uses your
    #  learned SVM classifier to determine whether the email is Spam or
    #  Not Spam

    # Set the file to be read in (change this to spamSample2.txt,
    # emailSample1.txt or emailSample2.txt to see different predictions on
    # different emails types). Try your own emails as well!
    def test_try_your_own_email(self):
        file_name_list = [('spamSample1.txt', 0), ('spamSample2.txt', 1), ('emailSample1.txt', 0), ('emailSample2.txt', 0)]

        for f, ret in file_name_list:
            with open('resource/' + f, 'r') as emailSample:
                file_contents = emailSample.read()
                from ex6_SVM.processEmail import processEmail
                word_indices = processEmail(file_contents, self.vocabList)
                from ex6_SVM.emailFeatures import emailFeatures
                x = emailFeatures(word_indices)
                p = self.model.predict(x.flatten().reshape(1, -1))
                print('Test Accuracy: {accuracy}'.format(accuracy=p[0]))
                self.assertEqual(p[0], ret)
