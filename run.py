import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

classifiers = [
	(SVC(kernel="linear", C=0.5), 'SVM'), 
	(DecisionTreeClassifier(max_leaf_nodes=3), 'Decision Tree'),
	(MLPClassifier(alpha=0.5, max_iter=2000), 'Neural Network'),
	(GaussianNB(), 'Naive Bayes'),
	]

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

for clf, clf_name in classifiers:
    cv_scores =  cross_val_score(clf, iris.data, iris.target, cv=10)
    print('[{}] CV accuracy: {:.2f}'.format(clf_name, cv_scores.mean()))
