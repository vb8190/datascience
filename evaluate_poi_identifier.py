#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import cross_validation
import pandas as pd

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

feature_train,feature_test,label_train,label_test = cross_validation.train_test_split(features,labels, test_size = 0.30, random_state = 42)

clf = tree.DecisionTreeClassifier()
clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)
acc = accuracy_score(pred,label_test)

print acc

print "Total pred......",int(sum(pred))


print (classification_report (label_test, pred ))
print (confusion_matrix (label_test , pred))


print len(feature_test)

p_score = precision_score(label_test , pred)

r_score = recall_score(label_test , pred)

print "precision score ",p_score
print "recall score ",r_score

pred = [0.] * 29
#print pred
accuracy = accuracy_score(pred, label_test)
print accuracy


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 

true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print (confusion_matrix (true_labels , predictions))
### your code goes here 

#FP = 3
#TP = 6
#FN = 2
#TN = 9

print "precision = ",precision_score(true_labels, predictions)
#precision = 0.666666666667
print "recall = ",recall_score(true_labels, predictions)
#recall = 0.75
