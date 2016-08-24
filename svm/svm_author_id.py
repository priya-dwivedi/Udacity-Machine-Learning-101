#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
import os
os.chdir(r"C:\Users\s6324900\Documents\ud120projects\tools")
   
import sys
from time import time
#sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100] 

from sklearn.svm import SVC
clf = SVC(C=10000.0, kernel = "rbf")

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print pred[10]
print pred[26]
print pred[50]

print sum(pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)

print acc


#########################################################
### your code goes here ###

#########################################################


