import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.datasets import load_iris

#cargar el modelo previamente entrenado
model = load_model('iris_model.h5')

#cargar los datos de iris

iris = load_iris()
class_names=iris.target_names

# configuración de la aplicación streamlit
st.title('Clasificación de flores Iris')
st.write('Esta aplicación predice la clase de una flor Iris.')

# selección de la especie de flor
sepal_length = st.slider("longitud del sépalo",4.0,8.0,5.0)
sepal_width =  st.slider("Ancho del sépalo",2.0,4.5,3.0)
petal_length = st.slider("longitud del pétalo",1.0,7.0,1.5)
petal_width = st.slider("Ancho del pétalo",0.1,2.5,0.2)
#Botón para predecir
if st.button('Predecir'):
    # crear un arreglo con los datos de la flor
    X = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    # predecir la clase de la flor
    prediction = model.predict(X)
    prediction_class=class_names[np.argmax(prediction)]
    st.write(f'La especie de la flor es: {prediction_class}')