from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle

arr = pd.read_excel('home.xlsx')
scaler = preprocessing.StandardScaler()
X = arr[['x0', 'x1', 'x2']]
X_normalized = scaler.fit_transform(X)
y = arr['y']
model = LinearRegression()
model.fit(X_normalized, y)
with open('home', 'wb') as pkl:
    pickle.dump(model, pkl)
