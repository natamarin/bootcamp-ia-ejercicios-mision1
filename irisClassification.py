from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos Iris
iris = load_iris()

# Convertir a dataframe para manipulación y visualización
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

# Exploración de datos
# Añadimos nombre de clases para mejorar compresión visual
df['species'] = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
# Graficar caracteristicas
sns.pairplot(df, hue='species')
plt.show()

# Grafica de caja
# Configurar tamaño de la figura
plt.figure(figsize=(12, 8))
# Gráfica de caja para cada característica agrupada por especie
for i, column in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=column, data=df)
plt.tight_layout
plt.show()