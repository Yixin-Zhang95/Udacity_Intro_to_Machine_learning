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

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

clf = DecisionTreeClassifier()
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)



# How many POIs are predicted for the test set for your POI identifier?
import numpy as np
print(np.array(labels_test))
print(len([e for e in labels_test if e == 1.0]))

#How many people total are in your test set?
print(len(features_test))



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(labels_test, pred, target_names=features_list))
print(confusion_matrix(labels_test, pred))


# What is the precision?
print("What is the precision?")
from sklearn.metrics import *
print(precision_score(labels_test, pred))


# What is the recall?
print("What is the recall?")
print(recall_score(labels_test, pred))

# Here are some made-up predictions and true labels for a hypothetical test set; fill in the following boxes to practice identifying true positives, false positives, true negatives, and false negatives. Let’s use the convention that “1” signifies a positive result, and “0” a negative.
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
tp = 0
tn = 0
fp = 0
fn = 0
for i, j in zip(predictions, true_labels):
    if i == 1 and j == 1:
        tp += 1
    elif i == 0 and j == 0:
        tn += 1
    elif i == 1 and j == 0:
        fp += 1
    elif i == 0 and j == 1:
        fn += 1
precision = tp/(tp + fp)
recall = tp/(tp + fn)
print('true positive is {}'.format(tp))
print('true negative is {}'.format(tn))
print('false positive is {}'.format(fp))
print('false negative is {}'.format(fn))
print('precision is {}'.format(precision))
print('recall is {}'.format(recall))


