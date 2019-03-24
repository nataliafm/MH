# -*- coding: utf-8 -*-
"""
@author: Natalia Fernández Martínez
"""

from sklearn.neighbors import KNeighborsClassifier
from scipy.io import arff #para leer los ficheros de datos
import numpy as np
from sklearn.model_selection import StratifiedKFold

#Función que recibe los datos leídos de los ficheros y los divide en X e y,
#   y normaliza los valores de X para que estén todos dentro del intervalo [0,1]
def tratamientoDatos(datos):
    X = []
    y = []
    
    #separar X e y
    for i in datos:
        i = i.tolist()
        X.append(i[:-1])
        y.append(i[-1])

    #normalizar los valores de X
    for i in range(len(X)):
        X[i] = list(X[i])
        maximo = -1000.0
        minimo = 1000.0
        
        for j in X[i]:
            if j > maximo: maximo = j
            if j < minimo: minimo = j
        
        for j in range(len(X[i])):
            X[i][j] = (X[i][j] - minimo) / (maximo - minimo)
    
    return X,y

#Función que recibe X e y, los divide en 5 secciones y hace 5-fold cross 
#   validation con ellos.
def KNN(X, y):
    #Crear la división en 5 secciones
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(X,y)

    porcentajes = []
    #Iterar sobre las 5 secciones distintas
    for i, j in skf.split(X,y):
        num_aciertos = 0
        
        #Separar los datos de entrenamiento y prueba en listas distintas
        entrenamientox = [X[k] for k in i]
        entrenamientoy = [y[k] for k in i]
        
        pruebax = [X[k] for k in j]
        pruebay = [y[k] for k in j]
        
        #Crear el clasificador para que sea 1NN con vector de pesos que le
        #   da la misma importancia a todas las características
        clasificador = KNeighborsClassifier(n_neighbors=1, weights='uniform')
        
        #Entrenar al clasificador con los datos de entrenamiento
        clasificador.fit(entrenamientox, entrenamientoy)
        
        #Probar el clasificador con los datos de prueba, llevando la cuenta de
        #   los valores que se predicen correctamente
        for k in range(len(pruebax)):
            pred = clasificador.predict([pruebax[k]])
            if pruebay[k] == pred: num_aciertos += 1
        
        #Guarda el porcentaje de aciertos sobre el número de valores de prueba
        porcentajes.append(100 * num_aciertos / len(pruebax))
        
    #Devuelve los porcentajes obtenidos en los 5 cálculos
    return porcentajes

#Función que encuentra el amigo más cercano dentro de un conjunto al elemento
#   del mismo con tiene índice i
def encontrarAmigo(conjunto, i, valores):
    sol = []
    val_sol = 1000000.0
    
    #leave one out
    aux = np.delete(conjunto, i, 0)
    val_aux = np.delete(valores, i, 0)
    
    val_i = sum(conjunto[i]) #suma de las características del elemento
    for j in range(len(aux)):
        suma = sum(aux[j])
        distancia = (val_i - suma)**2 #distancia euclídea
        
        #si es el más cercano y son amigos, puede ser la solución
        if distancia < val_sol and val_aux[j] == valores[i]:
            sol = j
    
    return conjunto[sol]

#Función que encuentra el enemigo más lejano dentro de un conjunto al elemento
#   del mismo con tiene índice i
def encontrarEnemigo(conjunto, i, valores):
    sol = []
    val_sol = 1000000.0
    aux = np.copy(conjunto)
    
    val_i = sum(conjunto[i]) #suma de las características del elemento
    for j in range(len(aux)):
        suma = sum(aux[j])
        distancia = (val_i - suma)**2 #distancia euclídea
        
        #si es el más cercano y son enemigos, puede ser la solución
        if distancia < val_sol and valores[j] != valores[i]:
            sol = j
    
    return conjunto[sol]
    

def RELIEF(X,y):
    #Crear la división en 5 secciones
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(X,y)

    porcentajes = []
    #Iterar sobre las 5 secciones distintas
    for i, j in skf.split(X,y):
        num_aciertos = 0
        
        #Separar los datos de entrenamiento y prueba en listas distintas
        entrenamientox = [X[k] for k in i]
        entrenamientoy = [y[k] for k in i]
        
        pruebax = [X[k] for k in j]
        pruebay = [y[k] for k in j]
    
        #Inicializar el vector de pesos con ceros
        w = np.zeros(len(entrenamientox[0]))
        entrenamientox = np.asarray(entrenamientox)
        
        #Para cada elemento del conjunto de entrenamiento, modifica el vector
        # de pesos sumándole el vector con la diferencia entre el elemento y 
        # su enemigo más cercano, y restándole el vector con la diferencia
        # entre el elemento y su amigo más cercano
        for k in range(len(entrenamientox)):
            amigo = np.asarray(encontrarAmigo(entrenamientox, k, entrenamientoy))
            enemigo = np.asarray(encontrarEnemigo(entrenamientox, k, entrenamientoy))
            
            w = w + (entrenamientox[k] - enemigo) - (entrenamientox[k] - amigo)
        
        #Se obtiene el mayor peso y se usa para normalizar el resto de elementos
        #   del vector. Los elementos menores que 0 se truncan a 0
        wm = max(w)
        for k in range(len(w)):
            if w[k] < 0.0: w[k] = 0.0
            else: w[k] = w[k] / wm
        
        #Multiplicar los inputs por los pesos obtenidos
        entrenamientox *= w
        pruebax *= w
        
        #Entrenar a un clasificador con los datos de entrenamiento, y obtener
        #   las predicciones para los datos de prueba
        clasificador = KNeighborsClassifier(n_neighbors=1)
        clasificador.fit(entrenamientox, entrenamientoy)
        
        for k in range(len(pruebax)):
            pred = clasificador.predict([pruebax[k]])
            if pruebay[k] == pred: num_aciertos += 1
        
        porcentajes.append(100 * num_aciertos / len(pruebax))
            
    return porcentajes

#lectura de los ficheros de datos
datos1, meta1 = arff.loadarff('datos/colposcopy.arff')
datos2, meta2 = arff.loadarff('datos/ionosphere.arff')
datos3, meta3 = arff.loadarff('datos/texture.arff')

#colposcopy
X, y = tratamientoDatos(datos1)
porcentajes = KNN(X,y)
print("Porcentaje 1NN: ", sum(porcentajes)/5.0)

porcentajes = RELIEF(X,y)
print("Porcentaje RELIEF: ", sum(porcentajes)/5.0)

#ionosphere
X, y = tratamientoDatos(datos2)
porcentajes = KNN(X,y)
print("Porcentaje 1NN: ", sum(porcentajes)/5.0)

porcentajes = RELIEF(X,y)
print("Porcentaje RELIEF: ", sum(porcentajes)/5.0)

#texture
X, y = tratamientoDatos(datos3)
porcentajes = KNN(X,y) 
print("Porcentaje 1NN: ", sum(porcentajes)/5.0)

porcentajes = RELIEF(X,y)
print("Porcentaje RELIEF: ", sum(porcentajes)/5.0)





