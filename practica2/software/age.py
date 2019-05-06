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

def Cruce_BLX(alfa, cromosoma1, cromosoma2):
    hijo1 = []
    hijo2 = []
    for i in range(len(cromosoma1)):
        Cmax, Cmin = 0, 0
        if cromosoma1[i] > cromosoma2[i]: 
            Cmax = cromosoma1[i]
            Cmin = cromosoma2[i]
        else: 
            Cmax = cromosoma2[i]
            Cmin = cromosoma1[i]
            
        I = Cmax - Cmin
        val = np.random.uniform(Cmin-I*alfa, Cmax+I*alfa)
        if val < 0.0: val = 0.0
        if val > 1.0: val = 1.0
        hijo1.append(val)
        
        val = np.random.uniform(Cmin-I*alfa, Cmax+I*alfa)
        if val < 0.0: val = 0.0
        if val > 1.0: val = 1.0
        hijo2.append(val)
    
    return hijo1, hijo2

def Cruce_aritmetico(cromosoma1, cromosoma2):
    hijo1 = []
    hijo2 = []
    for i in range(len(cromosoma1)):
        hijo1.append(0.7 * cromosoma1[i] + 0.3 * cromosoma2[i])
        hijo2.append(0.3 * cromosoma1[i] + 0.7 * cromosoma2[i])
    
    return hijo1, hijo2

def AGE(X, y, skf, tipo):
    porcentajes = []
    reduccion = []
    tiempos = []
    
    tam_poblacion = 30
    prob_mutacion = 0.001
    pm_cromosoma = prob_mutacion * len(X[0])
    
    for i, j in skf.split(X,y):
        #División en entrenamiento y prueba de las particiones
        entrenamientox = [X[k] for k in i]
        entrenamientoy = [y[k] for k in i]
        
        pruebax = [X[k] for k in j]
        pruebay = [y[k] for k in j]
        
        num_train = int(0.8 * len(entrenamientox))
        
        start = time.time()
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
            trainx = entrenamientox[:num_train]
            trainy = entrenamientoy[:num_train]
            
            testx = entrenamientox[num_train:]
            testy = entrenamientoy[num_train:]
            
            trainx *= w
            #entrenamientox_aux = entrenamientox * np.asarray(w)
                    
            clasificador = KNeighborsClassifier(n_neighbors=1)
            clasificador.fit(trainx, trainy)
            
            pred = clasificador.predict(testx)
                
            num_aciertos = 0
            tasa_cas = 0.0
            nulos = 0
            tasa_red = 0.0
                
            num_aciertos = len([b for a, b in enumerate(testy) if b == pred[a]])
            tasa_cas = 100 * num_aciertos / len(testx)

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
            # modelo estacionario --> elegir dos padres
            padres = []
            for i in range(2):
                val1 = np.random.choice(indices)
                uno = poblacion[val1]
                
                val2 = np.random.choice(indices)
                dos = poblacion[val2]
                
                if valores[val1] > valores[val2]:
                    padres.append(uno)
                else:
                    padres.append(dos)
            
            #Cruces
            padres_cruzados = []
            if tipo == 0:
                h1, h2 = Cruce_BLX(0.3, padres[0], padres[1])
                padres_cruzados.append(h1)
                padres_cruzados.append(h2)
            else:
                h1, h2 = Cruce_aritmetico(padres[0], padres[1])
                padres_cruzados.append(h1)
                padres_cruzados.append(h2)
            
            #Mutaciones
            poblacion_mutada = padres_cruzados
            
            for k in range(2):
                prob = np.random.random()
                if prob < pm_cromosoma:
                    cromosoma = np.random.randint(0, len(poblacion_mutada))
                    gen = np.random.randint(0, len(poblacion_mutada[0]))
                
                    Z = float(np.random.normal(0.0, 0.3, 1))
                    poblacion_mutada[cromosoma][gen] += Z
            
            #Evaluar los dos hijos creados
            valores_hijos = []
            for i in range(2):
                #Evaluar el cromosoma
                trainx = entrenamientox[:num_train]
                trainy = entrenamientoy[:num_train]
                
                testx = entrenamientox[num_train:]
                testy = entrenamientoy[num_train:]
                
                trainx *= np.asarray(poblacion_mutada[i])
                #entrenamientox_aux = entrenamientox * np.asarray(poblacion_mutada[i])
                #pruebax_aux = pruebax * np.asarray(poblacion_mutada[i])
                        
                clasificador = KNeighborsClassifier(n_neighbors=1)
                clasificador.fit(trainx, trainy)
                
                pred = clasificador.predict(testx)
                    
                num_aciertos = 0
                tasa_cas = 0.0
                nulos = 0
                tasa_red = 0.0
                    
                num_aciertos = len([b for a, b in enumerate(testy) if b == pred[a]])
                tasa_cas = 100 * num_aciertos / len(testx)
    
                nulos = len([a for a in w if a <= 0.2]) 
                tasa_red = 100 * nulos / len(w)
                    
                #Evaluación de la función objetivo
                valor = 0.5 * tasa_cas + 0.5 * tasa_red
                num_evaluaciones += 1
                
                valores_hijos.append(valor)
            
            #Obtener los 2 peores miembros de la población anterior y sustituirlos
            #   si son peores que los hijos obtenidos
            for i in range(2):
                index_peor = valores.index(min(valores))
                if valores[index_peor] < valores_hijos[i]:
                    poblacion[index_peor] = poblacion_mutada[i]
                    valores[index_peor] = valores_hijos[i]
            '''
            #Evaluar la nueva población
            valores = []
            for k in range(tam_poblacion):
                #Evaluar el cromosoma
                #print((poblacion[k]))
                entrenamientox_aux = entrenamientox * np.asarray(poblacion[k])
                        
                clasificador = KNeighborsClassifier(n_neighbors=1)
                clasificador.fit(entrenamientox_aux, entrenamientoy)
                
                pred = clasificador.predict(entrenamientox)
                    
                num_aciertos = 0
                tasa_cas = 0.0
                nulos = 0
                tasa_red = 0.0
                    
                num_aciertos = len([b for a, b in enumerate(entrenamientoy) if b == pred[a]])
                tasa_cas = 100 * num_aciertos / len(entrenamientox)
    
                nulos = len([a for a in w if a <= 0.2]) 
                tasa_red = 100 * nulos / len(w)
                    
                #Evaluación de la función objetivo
                valor = 0.5 * tasa_cas + 0.5 * tasa_red
                num_evaluaciones += 1
                
                valores.append(valor)
            '''
            
        #Obtener el mejor valor de la población obtenida
        w = poblacion[valores.index(max(valores))]
        entrenamientox *= np.asarray(w)
        #pruebax_aux = pruebax * np.asarray(w)
        
        clasificador = KNeighborsClassifier(n_neighbors=1)
        clasificador.fit(entrenamientox, entrenamientoy)
        
        pred = clasificador.predict(pruebax)
        
        #Evaluar el vector de pesos obtenido
        num_aciertos = len([b for a, b in enumerate(pruebay) if b == pred[a]])
        tasa_cas = 100 * num_aciertos / len(pruebax)
    
        nulos = len([a for a in w if a <= 0.2]) 
        tasa_red = 100 * nulos / len(w)
        
        porcentajes.append(tasa_cas)
        reduccion.append(tasa_red)
        end = time.time()
        tiempos.append(end-start)
        print(tasa_cas)
        
    return porcentajes, reduccion, tiempos

#lectura de los ficheros de datos
datos1, meta1 = arff.loadarff('datos/colposcopy.arff')
datos2, meta2 = arff.loadarff('datos/ionosphere.arff')
datos3, meta3 = arff.loadarff('datos/texture.arff')

#colposcopy
X, y, skf = tratamientoDatos(datos1)
porcentajes, reduccion, tiempos = AGE(X, y, skf, 0)
print("Porcentajes AGE con BLX: ", porcentajes)
print("Reducción AGE con BLX: ", reduccion)
print("Tiempos: ", tiempos)

porcentajes, reduccion, tiempos = AGE(X, y, skf, 1)
print("Porcentajes AGE con AC: ", porcentajes)
print("Reducción AGE con AC: ", reduccion)
print("Tiempos: ", tiempos)
print('\n')

#ionosphere
X, y, skf = tratamientoDatos(datos2)
porcentajes, reduccion, tiempos = AGE(X, y, skf, 0)
print("Porcentajes AGE con BLX: ", porcentajes)
print("Reducción AGE con BLX: ", reduccion)
print("Tiempos: ", tiempos)

porcentajes, reduccion, tiempos = AGE(X, y, skf, 1)
print("Porcentajes AGE con AC: ", porcentajes)
print("Reducción AGE con AC: ", reduccion)
print("Tiempos: ", tiempos)
print('\n')
#texture
X, y, skf = tratamientoDatos(datos3)
porcentajes, reduccion, tiempos = AGE(X, y, skf, 0)
print("Porcentajes AGE con BLX: ", porcentajes)
print("Reducción AGE con BLX: ", reduccion)
print("Tiempos: ", tiempos)

porcentajes, reduccion, tiempos = AGE(X, y, skf, 1)
print("Porcentajes AGE con AC: ", porcentajes)
print("Reducción AGE con AC: ", reduccion)
print("Tiempos: ", tiempos)