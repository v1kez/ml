import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

lapa_df = pd.read_excel('lapa.xlsx')
lapa_df
label_encoder=LabelEncoder()
lapa_df["y"]=label_encoder.fit_transform(lapa_df["y"])
X=lapa_df.drop(["y"],axis=1)
Y=lapa_df["y"]
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.2,random_state=3)
model=LogisticRegression()
model.fit(X_train1,Y_train1)
with open('lapa_log', 'wb') as pkl:
    pickle.dump(model, pkl)
