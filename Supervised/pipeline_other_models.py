import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.manifold import TSNE
from utils import *
import random


random.seed(1616514654156435)
# Parametros SVC
# kernel
# 'linear'
# 'rbf', gamma='scale',
# 'polynomial', d = grado polinomio

# Definición datos
dataset = load_breast_cancer( as_frame=True )
x = dataset["data"].values
x = (x - x.mean()) / x.std()
y = dataset["target"]

# Separación train, validation y test
x_train, x_val_test, y_train, y_val_test = train_test_split( x, y, test_size=0.4, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split( x_val_test, y_val_test, test_size=0.5, shuffle=True)


models = [LR(), SVC(kernel="rbf", gamma="scale"), SVC(kernel="linear"),
        SVC(kernel="poly"), DecisionTreeClassifier()]
models_names = ["LR", "SVC radial", "SVC linear", "SVC Polynomial", "Tree"]
models_metrics_train = {}
models_metrics_val = {}
for name, model in zip(models_names, models):
    # Entrenar el modelo con los datos en altas dimensiones
    model.fit(x_train, y_train)

    # Entrenar el modelo con los datos en dimensiones reducidas
    # tsne =  TSNE(n_components=3, init="pca").fit(x_train)
    # x_train_embedded = tsne.transform(x_train)
    # reduced_model = clone(model)
    # reduced_model.fit(x_train_embedded, y_train)


    # Realizar la comparación del modelo del paso 8 contra el modelo del paso
    y_train_pred = model.predict(x_train)
    models_metrics_train[name] = get_metrics(y_train, y_train_pred)
    y_val_pred = model.predict(x_val)
    models_metrics_val[name] = get_metrics(y_val, y_val_pred)

print("-------- Train ---------")
print(pd.DataFrame(models_metrics_train))
print("-------- Valid ---------")
print(pd.DataFrame(models_metrics_val))
