import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

iris = pd.read_csv('./Iris.csv')

# model 0
print("Model 0 : All")
recTime = time.time()
X = iris.drop('Species', axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svm_model = svm.SVC(kernel = "linear")
svm_model.fit(X_train, y_train)
print(classification_report(y_test, svm_model.predict(X_test)))
print("time: ", time.time() - recTime)

# model 1
print("Model 1: PetalLengthCm  ,PetalWidthCm")
recTime = time.time()
X = iris.loc[:,["PetalLengthCm","PetalWidthCm"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svm_model = svm.SVC(kernel = "linear")
svm_model.fit(X_train, y_train)
print(classification_report(y_test, svm_model.predict(X_test)))
print("time: ", time.time() - recTime)

# model 2
print("Model 2: PetalLengthCm , SepalLengthCm")
recTime = time.time()
X = iris.loc[:,["PetalLengthCm","SepalLengthCm"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svm_model = svm.SVC(kernel = "linear")
svm_model.fit(X_train, y_train)
print(classification_report(y_test, svm_model.predict(X_test)))
print("time: ", time.time() - recTime)