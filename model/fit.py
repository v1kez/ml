import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data_df = pd.read_excel('lapa.xlsx')
label_encoder = LabelEncoder()
data_df["y"]=label_encoder.fit_transform(data_df["y"])
X=data_df.drop(["y"],axis=1)
Y=data_df["y"]
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.3,random_state=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train1, Y_train1)
y_pred = model.predict(X_test1)
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.5,random_state=10)
y_true = Y_train1
y_pred = Y_test1
target_names = ['class 0', 'class 1']
acc=classification_report(y_true, y_pred, target_names=target_names)
print(acc)
df = pd.DataFrame({'y_pred': y_pred,
                   'Y_test1': Y_test1})
with open('lapa_fileKNN', 'wb') as pkl:
    pickle.dump(model, pkl)
