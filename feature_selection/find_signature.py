#!/usr/bin/python3

import joblib
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"

word_data = joblib.load( open(words_file, "rb"))
authors = joblib.load( open(authors_file, "rb"))



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]



### your code goes here
# from sklearn import tree
# from sklearn.metrics import accuracy_score
# clf = tree.DecisionTreeClassifier().fit(features_train, labels_train)
# pred = clf.predict(features_test)
# score = accuracy_score(labels_test, pred)
# print('accuracy is {}'.format(score))

from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)
print(acc)
# 0.9476678043230944

# iterate through the feature importance list
importances = clf.feature_importances_
for index, item in enumerate(importances):
    if item > 0.2:        
        print(index, item)
# 33614 0.7647058823529412

# which word causes this feature?
print(vectorizer.get_feature_names()[33614])
# stephanlonect

# remove this word in vectorize.py

# run find_signature.py again, now another powerful word occurs in position 14343
print(vectorizer.get_feature_names()[14343])
# cgermannsf

# remove this word in vectorize.py

# run find_signature.py again, now there is another powerful word, but important is only 0.36363636363636365, so keep it
# now the accuracy is 0.8168373151308305, a more reasonable number