#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
feature_train,feature_test,label_train,label_test = cross_validation.train_test_split(features,labels, test_size = 0.30, random_state = 42)

clf = tree.DecisionTreeClassifier()
clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)
acc = accuracy_score(pred,label_test)

print acc

### it's all yours from here forward!  



