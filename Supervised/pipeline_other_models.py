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


def normalize( x ):
    return (x - x.mean()) / x.std()

def latex_with_lines(df, *args, **kwargs):
    kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels
                                            + ['r'] * df.shape[1] + [''])
    res = df.to_latex(*args, **kwargs)
    return res.replace('\\\\\n', '\\\\ \\midrule\n')

np.random.seed(1234567)
# Parametros SVC
# kernel
# 'linear'
# 'rbf', gamma='scale',
# 'polynomial', d = grado polinomio

# Definición datos
dataset = load_breast_cancer( as_frame=True )
x,y = dataset["data"], dataset["target"]
# x = TSNE(n_components=3, init="pca").fit_transform(x)
x,y = normalize( x ), y #classes2binary( y.values )
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, shuffle=True )



models = [LR(), SVC(kernel="rbf", gamma="scale"), SVC(kernel="linear"),
        SVC(kernel="poly"), DecisionTreeClassifier()]
models_names = ["LR", "SVC radial", "SVC linear", "SVC Polynomial", "Tree"]
models_metrics_train = {}
models_metrics_test = {}
for name, model in zip(models_names, models):
    # Entrenar el modelo con los datos en altas dimensiones
    model.fit(x_train, y_train)

    # Realizar la comparación del modelo del paso 8 contra el modelo del paso
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    if hasattr(model, 'predict_proba'):
        y_prob_train = model.predict_proba(x_train)[:, 1]
        y_prob_test = model.predict_proba(x_test)[:, 1]
    else:
        y_prob_train = None
        y_prob_test = None
        print(f"{name} tiene predict proba ")
    models_metrics_train[name] = get_metrics(y_train, y_train_pred, y_prob_train)
    models_metrics_test[name] = get_metrics(y_test, y_test_pred, y_prob_test)

print("-------- Train ---------")
print(pd.DataFrame(models_metrics_train))
print("-------- Valid ---------")
print(pd.DataFrame(models_metrics_test))

latex_repr = latex_with_lines(pd.DataFrame(models_metrics_test).T)
print(latex_repr)