from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle

lapa = pd.read_excel('lapa.xlsx')
X=lapa.drop(["y"],axis=1)
Y=lapa["y"]
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,random_state=3)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train1, Y_train1)
with open('Lapa_drevo', 'wb') as pkl:
    pickle.dump(model, pkl)
