import pickle
import numpy as np
from flask import Flask, render_template, url_for, request

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


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторная работа №10", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод KNN", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           ]])
        pred = loaded_model_knn.predict(X_new)
        animal = animal_dict[pred[0]]
        return render_template('lab1.html', title="Метод KNN", menu=menu,
                               class_model="Это: " + animal)

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu)
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           ]])
        pred = loaded_model_Log.predict(X_new)
        animal = animal_dict[pred[0]]

        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + animal)

@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Дерево решений", menu=menu)
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           ]])
        pred = loaded_model_Tree.predict(X_new)
        animal = animal_dict[pred[0]]
        return render_template('lab3.html', title="Дерево решений", menu=menu,
                               class_model="Это: " + animal)


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
