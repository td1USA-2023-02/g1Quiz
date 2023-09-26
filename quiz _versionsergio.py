import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

pd.set_option("display.max_columns",None)

iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

print(df.head())
print(df.info())


def asignar_etiqueta(row):
    condiciones = [
        row['sepal length (cm)'] >= 5.1,
        row['sepal width (cm)'] >= 3.5,
        row['petal length (cm)'] >= 1.3,
        row['petal width (cm)'] <= 0.2
    ]
    return 'margarita' if all(condiciones) else 'no es margarita'


df['clasificado'] = df.apply(asignar_etiqueta, axis=1)

print(df.head())

##Cargar csv

# Ruta al archivo CSV de transformado
archivo_transformado = "iris_transformado.csv"

# Guardar los datos transformados en un nuevo archivo CSV
df.to_csv(archivo_transformado, index=False)

# Verificar que se haya guardado correctamente
print(f"Los datos transformados se han guardado en {archivo_transformado}")
