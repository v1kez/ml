import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

animal_dict = {
    0: "Кот",
    1: "Собака"
}



menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        # {"name": "Лаба 4", "url": "p_lab4"}
        ]

loaded_model_knn = pickle.load(open('model/lapa_fileKNN', 'rb'))
loaded_model_Log = pickle.load(open('model2/lapa_log', 'rb'))
loaded_model_Tree = pickle.load(open('model3/Lapa_drevo', 'rb'))
# loaded_model_home = pickle.load(open('model4/home', 'rb'))

def classification_model_metrics(model: str) -> dict:
    models = {"knn": loaded_model_knn, "logistic_regression": loaded_model_Log, "tree": loaded_model_Tree}

    model_selected = models[model]
    lapa = pd.read_excel('model/lapa.xlsx')
    lapa.drop_duplicates(inplace=True)

    label_encoder = LabelEncoder()
    lapa["y"] = label_encoder.fit_transform(lapa["y"])

    x = lapa.drop(["y"], axis=1)
    y = lapa["y"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

    model_selected.fit(x_train, y_train)
    y_pred = model_selected.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return {"accuracy": round(accuracy, 3), "precision": round(precision, 3), "recall": round(recall, 3)}



@app.route("/")
def index():
    return render_template('index.html', title="Лабораторную работу выполнил Буренок Д.Р.", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод KNN", menu=menu, class_model='')
    if request.method == 'POST':
        metrics = classification_model_metrics("knn")
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           ]])
        pred = loaded_model_knn.predict(X_new)
        animal = animal_dict[pred[0]]
        return render_template('lab1.html', title="Метод KNN", menu=menu,
                               class_model="Это: " + animal, accuracy=metrics['accuracy'],
                               precision=metrics['precision'], recall=metrics['recall'])

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu)
    if request.method == 'POST':
        metrics = classification_model_metrics("logistic_regression")
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           ]])
        pred = loaded_model_Log.predict(X_new)
        animal = animal_dict[pred[0]]

        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + animal, accuracy=metrics['accuracy'],
                               precision=metrics['precision'], recall=metrics['recall'])

@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Дерево решений", menu=menu)
    if request.method == 'POST':
        metrics = classification_model_metrics("tree")
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           ]])
        pred = loaded_model_Tree.predict(X_new)
        animal = animal_dict[pred[0]]

        return render_template('lab3.html', title="Дерево решений", menu=menu,
                               class_model="Это: " + animal, accuracy=metrics['accuracy'],
                               precision=metrics['precision'], recall=metrics['recall'])

@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('lapa_length')),
                       float(request.args.get('lapa_width'))]])
    pred = loaded_model_knn.predict(X_new)
    animal = animal_dict[pred[0]]
    return jsonify(sort=animal)


# @app.route("/p_lab4", methods=['POST', 'GET'])
# def f_lab4():
#     if request.method == 'GET':
#         return render_template('lab4.html', title="Регрессия", menu=menu)
#     if request.method == 'POST':
#         X_new = np.array([[int(request.form['list1']),
#                            int(request.form['list2']),
#                            int(request.form['list3']),
#                            ]])
#         pred = loaded_model_home.predict(X_new)
#         return render_template('lab4.html', title="Регрессия", menu=menu,
#                                class_model="Это: " + pred[0])

if __name__ == "__main__":
    app.run(debug=True)
