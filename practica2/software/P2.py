# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:47:20 2019

@author: natalia
"""
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import arff
import numpy as np
from sklearn.model_selection import StratifiedKFold
import time

np.random.seed(42)

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
    
    #Crear las particiones
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(X,y)
    
    return X, y, skf

#KNN con k=1 y pesos uniformes
def KNN(X, y, skf):
    porcentajes = []
    tiempos = []
    #Iterar sobre las 5 secciones distintas
    for i, j in skf.split(X,y):
        start = time.time()
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
        
        #Calcula el tiempo de ejecución de esa iteración
        end = time.time()
        tiempos.append(end-start)
        
    #Devuelve los porcentajes y los tiempos
    return porcentajes, tiempos

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
            
        val_sol = distancia
    
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
            
        val_sol = distancia
    
    return conjunto[sol]
    
#KNN con k=1 y pesos aprendidos usando el método greedy RELIEF
def RELIEF(X, y, skf):
    porcentajes = []
    reduccion = []
    tiempos = []
    #Iterar sobre las 5 secciones distintas
    for i, j in skf.split(X,y):
        start = time.time()
        num_ceros = 0
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
        
        #Obtener tasa de aciertos, tasa de reducción y tiempo de ejecución
        for k in range(len(pruebax)):
            pred = clasificador.predict([pruebax[k]])
            if pruebay[k] == pred: num_aciertos += 1
        
        porcentajes.append(100 * num_aciertos / len(pruebax))
        
        for k in w:
            if k < 0.2: num_ceros += 1
            
        reduccion.append(100*num_ceros/len(w))
        
        end = time.time()
        tiempos.append(end-start)
    
    return porcentajes, reduccion, tiempos

#KNN con k=1 y pesos aprendidos usando el método de búsqueda local best first
def BL(X, y, skf):
    #Inicializa el vector de pesos con valores aleatorios entre [0,1]
    w = np.random.rand(len(X[0]))
    
    porcentajes = []
    reduccion = []
    tiempos = []
    
    #Iterar sobre las 5 secciones distintas
    for i, j in skf.split(X,y):
        start = time.time()
        
        w = np.random.rand(len(X[0]))
        
        num_aciertos = 0
        num_ceros = 0
        
        #Separar los datos de entrenamiento y prueba en listas distintas
        entrenamientox = [X[k] for k in i]
        entrenamientoy = [y[k] for k in i]
        
        pruebax = [X[k] for k in j]
        pruebay = [y[k] for k in j]
        
        #Obtener valor del punto de partida
        entrenamientox_aux = entrenamientox * w
        pruebax_aux = pruebax * w
                
        clasificador = KNeighborsClassifier(n_neighbors=1)
        clasificador.fit(entrenamientox_aux, entrenamientoy)

        pred = clasificador.predict(pruebax_aux)
        num_aciertos = len([b for a, b in enumerate(pruebay) if b == pred[a]])
                
        tasa_cas = 100 * num_aciertos / len(pruebax_aux)
                
        nulos = 0
        nulos = len([a for a in w if a <= 0.2]) 
                    
        tasa_red = 100 * nulos / len(w)
                
        #Evaluación de la función objetivo
        valor_max = 0.5 * tasa_cas + 0.5 * tasa_red
        
        num_vecinos = 0
        iteraciones = 0
        
        #Exploración del vecindario
        while iteraciones < 30000:
            num_vecinos = 0
            w_original = w #se guarda el valor original del vector en caso de que tenga que volver atrás

            
            while num_vecinos < 20 * len(w):
                #Generación de un valor de la distribución normal
                Z = np.random.normal(0.0, 0.3, 1)
                
                #Elección aleatoria de un índice del vector de pesos
                k = np.random.choice(range(len(w)))
                
                #Se obtiene un vecino modificando el vector pesos y sumándole
                #   a uno de sus elementos un valor de la distribución normal
                valor_anterior = w[k]
                
                w[k] += Z
                if w[k] < 0.0: w[k] = 0.0
                if w[k] > 1.0: w[k] = 1.0
                
                #Si ha hecho 100% de reducción, deshace la modificación
                num_ceros = len(w) - np.count_nonzero(w)
                if num_ceros == len(w): w[k] = valor_anterior
                
                #Se comprueba si el nuevo vecino es mejor que el que ya tenemos
                entrenamientox_aux = entrenamientox * w
                pruebax_aux = pruebax * w
                
                clasificador = KNeighborsClassifier(n_neighbors=1)
                clasificador.fit(entrenamientox_aux, entrenamientoy)
                
                pred = clasificador.predict(pruebax_aux)
                
                num_aciertos = 0
                tasa_cas = 0.0
                nulos = 0
                tasa_red = 0.0
                
                num_aciertos = len([b for a, b in enumerate(pruebay) if b == pred[a]])

                tasa_cas = 100 * num_aciertos / len(pruebax_aux)

                nulos = len([a for a in w if a <= 0.2]) 
                    
                tasa_red = 100 * nulos / len(w)
                
                #Evaluación de la función objetivo
                valor = 0.5 * tasa_cas + 0.5 * tasa_red
                iteraciones += 1
                
                num_aciertos = 0
                nulos = 0
                #Si el valor para la función objetivo es mejor que el que tenemos,
                #   nos movemos a ese valor y ya no buscamos más vecinos
                if valor > valor_max:
                    valor_max = valor
                    break
                else: #Si el valor es peor, seguimos explorando el vecindario
                    w = w_original
                    num_vecinos += 1
                if iteraciones >= 15000: break
            
            if num_vecinos >= 20 * len(w): break
        
        #Multiplicar los inputs por los pesos obtenidos
        entrenamientox *= w
        pruebax *= w
        
        #Entrenar a un clasificador con los datos de entrenamiento, y obtener
        #   las predicciones para los datos de prueba
        clasificador = KNeighborsClassifier(n_neighbors=1)
        clasificador.fit(entrenamientox, entrenamientoy)
        
        #Obtener tasa de aciertos, tasa de reducción y tiempo de ejecución
        num_aciertos = 0
        num_ceros = 0
        for k in range(len(pruebax)):
            pred = clasificador.predict([pruebax[k]])
            if pruebay[k] == pred: num_aciertos += 1
        
        porcentajes.append(100 * num_aciertos / len(pruebax))
        
        for k in w:
            if k < 0.2: num_ceros += 1
        
        reduccion.append(100 * num_ceros / len(w))
        
        end = time.time()
        tiempos.append(end-start)
    
    return porcentajes, reduccion, tiempos

def Cruce_BLX(alfa, cromosoma1, cromosoma2):
    hijo = []
    for i in range(len(cromosoma1)):
        Cmax, Cmin = 0, 0
        if cromosoma1[i] > cromosoma2[i]: 
            Cmax = cromosoma1[i]
            Cmin = cromosoma2[i]
        else: 
            Cmax = cromosoma2[i]
            Cmin = cromosoma1[i]
            
        I = Cmax - Cmin
        
        hijo.append(np.random.uniform(Cmin-I*alfa, Cmax+I*alfa))
        

def AGG(X, y, skf):
    porcentajes = []
    reduccion = []
    tiempos = []
    
    tam_poblacion = 30
    prob_cruce = 0.7
    prob_mutacion = 0.001
    
    for i, j in skf.split(X,y):
        #División en entrenamiento y prueba de las particiones
        entrenamientox = [X[k] for k in i]
        entrenamientoy = [y[k] for k in i]
        
        pruebax = [X[k] for k in j]
        pruebay = [y[k] for k in j]
        
        #Crear la población y evaluarla
        poblacion = []
        valores = []
        indices = []
        for k in range(tam_poblacion):
            #Crear cromosoma
            w = np.random.rand(len(X[0]))
            poblacion.append(w)
            indices.append(k)
            
            #Evaluar el cromosoma
            entrenamientox_aux = entrenamientox * w
            pruebax_aux = pruebax * w
                    
            clasificador = KNeighborsClassifier(n_neighbors=1)
            clasificador.fit(entrenamientox_aux, entrenamientoy)
            
            pred = clasificador.predict(pruebax_aux)
                
            num_aciertos = 0
            tasa_cas = 0.0
            nulos = 0
            tasa_red = 0.0
                
            num_aciertos = len([b for a, b in enumerate(pruebay) if b == pred[a]])
            tasa_cas = 100 * num_aciertos / len(pruebax_aux)

            nulos = len([a for a in w if a <= 0.2]) 
            tasa_red = 100 * nulos / len(w)
                
            #Evaluación de la función objetivo
            valor = 0.5 * tasa_cas + 0.5 * tasa_red
            
            valores.append(valor)
            
        num_generacion = 0
        num_evaluaciones = 0
        
        while num_evaluaciones < 15000:
            num_generacion += 1
            
            #Selección (torneo binario)
            copia_poblacion = []
            while len(copia_poblacion) < tam_poblacion:
                val1 = np.random.choice(indices)
                uno = poblacion[val1]
                
                val2 = np.random.choice(indices)
                dos = poblacion[val2]
                
                if valores[val1] > valores[val2]:
                    copia_poblacion.append(uno)
                else:
                    copia_poblacion.append(dos)
            
            #Cálculo del número de padres que se cruzan
            num_cruzan =  int(prob_cruce * tam_poblacion)
            
            #Cruces
            poblacion_cruzada = []
            for k in range(num_cruzan):
                poblacion_cruzada.insert(Cruce_BLX(0.3, copia_poblacion[k], copia_poblacion[k+1]))
                k += 1
                
            
            
        
        

#lectura de los ficheros de datos
datos1, meta1 = arff.loadarff('datos/colposcopy.arff')
datos2, meta2 = arff.loadarff('datos/ionosphere.arff')
datos3, meta3 = arff.loadarff('datos/texture.arff')

#colposcopy
X, y, skf = tratamientoDatos(datos1)

#ionosphere
X, y, skf = tratamientoDatos(datos2)

#texture
X, y, skf = tratamientoDatos(datos3)

