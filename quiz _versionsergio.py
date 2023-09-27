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
#archivo_transformado = "iris_transformado.csv"
#archivo_transformado = "g1Quiz\iris_transformado.csv"
archivo_transformado = "iris_transformado.csv"
# Guardar los datos transformados en un nuevo archivo CSV
df.to_csv(archivo_transformado, index=False)


# Verificar que se haya guardado correctamente
print(f"Los datos transformados se han guardado en {archivo_transformado}")

print(df.head())
##EDA 



#Countplot
sns.countplot(df, x="clasificado").set(title="Cantidad de margaritas y no margaritas")
plt.show()


# Diagrama de caja de longitud del pétalo por especie
sns.boxplot(data=df, x="clasificado", y="petal length (cm)")
plt.title("Diagrama de Caja de Longitud del Pétalo")
plt.xlabel("clasificado")
plt.ylabel("Longitud del Pétalo (cm)")
plt.show()

# Diagrama de caja de longitud del sepalo por especie
sns.boxplot(data=df, x="clasificado", y="sepal length (cm)")
plt.title("Diagrama de Caja de Longitud del Sepalo")
plt.xlabel("clasificado")
plt.ylabel("Longitud del Sepalo (cm)")
plt.show()

# Diagrama de caja de longitud del sepalo por especie
sns.boxplot(data=df, x="clasificado", y="sepal width (cm)")
plt.title("Diagrama de Caja de Ancho del Sepalo")
plt.xlabel("clasificado")
plt.ylabel("Ancho del Sepalo (cm)")
plt.show()

# Diagrama de caja de longitud del sepalo por especie
sns.boxplot(data=df, x="clasificado", y="petal width (cm)")
plt.title("Diagrama de Caja de Ancho del Petalo")
plt.xlabel("clasificado")
plt.ylabel("Ancho del Petalo (cm)")
plt.show()


# Histograma de las longitudes del sépalo para cada clasificacion
sns.histplot(data=df, x="sepal length (cm)", hue="clasificado", bins=20)
plt.title("Histograma de Longitud del Sépalo")
plt.xlabel("Longitud del Sépalo (cm)")
plt.ylabel("Frecuencia")
plt.show()

# Diagrama de dispersión de longitud del sépalo vs. ancho del sépalo
sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)", hue="clasificado")
plt.title("Diagrama de Dispersión Sépalo")
plt.xlabel("Longitud del petalo (cm)")
plt.ylabel("Ancho del petalo (cm)")
plt.show()

##Concluir según resultados
    #La mayoría de las flores son no margaritas, hay aproximadamente 8 flores que son margaritas. 
    #Esta clasificación se debe a la longitud del petalo, donde las margaritas son de menor longitud de petalo,
    #tambien se puede observar que las margaritas son de mayor ancho de sepalo.
    #Y que la clasificación de no es margarita, tiene bastante desviación y 4 outliers. 
    #De acuerdo al ancho de petalo, las margaritas tienen menor ancho de petalo y los que no son margarita tienen bastante variación.
    #En el histograma según la longitud del sepalo se puede observar como las flores que son margaritas, tienen una longitud de aproximadamente 5 a 5.6.
    #