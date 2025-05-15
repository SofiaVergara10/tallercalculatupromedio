import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import sklearn 
import numpy as np

prom = pd.read_csv("datospromedio.csv")
st.title("Indicadores del Promedio de notas en la Universidad")
tab1, tab2, tab3 = st.tabs(["Análisis Univariado", "Análisis Bivariado", "Calcula tu Promedio"])

with open("modelo.pickle", "rb") as f:
    modelo = pickle.load(f)

with tab1:
    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    #promediouni
    ax[0].hist(prom["colGPA"])
    #edad
    ax[1].hist(prom["age"])
    #genero
    conteo = prom["male"].value_counts()
    ax[2].bar(conteo.index, conteo.values)
    #Acceso a PC
    conteo = prom["PC"].value_counts()
    ax[3].bar(conteo.index, conteo.values)
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    #edad vs promedionotas
    sns.scatterplot(data=prom, x="age", y="colGPA", ax=ax[0])
    #genero vs promedionotas
    sns.violinplot(data=prom, x="male", y="colGPA", ax=ax[1])
    #acceso a computador vs promedionotas
    sns.boxplot(data=prom, x="PC", y="colGPA", ax=ax[2])
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    age = st.slider("edad", 15, 100)
    male = st.selectbox("Género", ["Hombre", "Mujer"])
    if male == "Hombre":
        male = 1
    else:
        male = 0 
    PC = st.selectbox("Acceso_computador", ["Computador", "No_computador"])
    if PC == "Computador":
        PC = 1
    else:
        PC = 0
    if st.button("Predecir"):
        pred = modelo.predict(np.array([[age, male, PC]]))
        st.write(f"Su promedio sería {round(pred[0], 1)}")
