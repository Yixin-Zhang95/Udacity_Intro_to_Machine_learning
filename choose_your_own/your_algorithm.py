#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf_nb = GaussianNB().fit(features_train, labels_train)
pred_nb = clf_nb.predict(features_test)
accuracy_nb = accuracy_score(labels_test, pred_nb)
name = 'GaussianNB'
print('accuracy', accuracy_nb)


from sklearn import svm
clf_svm = svm.SVC().fit(features_train, labels_train)
pred_svm = clf_svm.predict(features_test)
accuracy_svm = accuracy_score(labels_test, pred_svm)
name = 'support_vector_machine'
print('accuracy', accuracy_svm)



from sklearn import tree
clf_dt = tree.DecisionTreeClassifier(min_samples_split = 40).fit(features_train, labels_train)
pred_dt = clf_dt.predict(features_test)
accuracy_dt = accuracy_score(labels_test, pred_dt)
name = 'decision_tree'
print('accuracy',accuracy_dt )

from sklearn.neighbors import KNeighborsRegressor
clf_kn = KNeighborsRegressor(n_neighbors=31).fit(features_train, labels_train)
pred_kn = clf_kn.predict(features_test)
# accuracy_kn = accuracy_score(labels_test, pred_kn)
name = 'knearest_neighbors'


from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(max_depth= None, random_state=0, min_samples_split =40).fit(features_train, labels_train)
pred_rf = clf_rf.predict(features_test)
accuracy_rf = accuracy_score(labels_test, pred_rf)
name = 'random_forest'
print('accuracy',accuracy_rf)




from sklearn.ensemble import AdaBoostClassifier
clf_ab = AdaBoostClassifier(n_estimators= 50, random_state=0).fit(features_train, labels_train)
pred_ab = clf_ab.predict(features_test)
accuracy_ab = accuracy_score(labels_test, pred_ab)
name = 'adaboost'
print('accuracy',accuracy_ab)


try:
    prettyPicture(clf_ab, features_test, labels_test, name)
except NameError as e:
    print(e)
