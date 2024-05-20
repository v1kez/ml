from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle

X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.5,random_state=10)
y_true = Y_train1
y_pred = Y_test1
target_names = ['class 0', 'class 1']
acc=classification_report(y_true, y_pred, target_names=target_names)
