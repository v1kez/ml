from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

lapa = pd.read_excel('lapa.xlsx')
label_encoder = LabelEncoder()
lapa["y"]=label_encoder.fit_transform(lapa["y"])
X=lapa.drop(["y"],axis=1)
Y=lapa["y"]
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,random_state=3)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train1, Y_train1)
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.5,random_state=10)
y_true = Y_train1
y_pred = Y_test1
target_names = ['class 0', 'class 1']
acc=classification_report(y_true, y_pred, target_names=target_names)
print(acc)

with open('Lapa_drevo', 'wb') as pkl:
    pickle.dump(model, pkl)
