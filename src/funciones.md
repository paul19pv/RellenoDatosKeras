

```python
# librerias para el procesamiento
import pandas as pd
import numpy as np
import datetime
from pylab import *
# libreria para generar el escalar
from sklearn.preprocessing import StandardScaler
#Librerias para red neuronal
from keras.models import Sequential
from keras.layers import Dense
from keras.constraints import min_max_norm
#librearia para calcular la correlacion
import seaborn as sns
```

    Using TensorFlow backend.
    


```python
#### Leer los datos de las estaciones

def leer_archivo(archivo,todas_estaciones):
    datos = pd.DataFrame()
    # leer datos de cada hoja de excel y asignarlos en una variable
    # index_col = ingresar el numero de columna de la fecha. Se cuenta desde cero. La columna fecha se convierte en el indice de la tabla
    # parsedates = para el ejemplo debe ser True. Analiza si el indice de la tabla es una fecha
    # sheet_name = Nombre de la hoja de calculo
    
    for i in todas_estaciones:
        # asignar en una variable los datos de la estacion
        datos_estacion = pd.read_excel(io=archivo,index_col=2,parsedates=True,sheet_name=i)
        # convertir la columna fecha al formato fecha 'AAAA-MM-DD'
        datos_estacion.index = pd.to_datetime(datos_estacion.index)
        # ordenar los datos por la fecha, desde más antigua
        datos_estacion['VALOR'].sort_index(inplace=True)
        # unir los datos de las estaciones en un solo DataFrame
    
        
        datos[i] = datos_estacion['VALOR']
    #### Anexar la columna semana a la matriz de datos
    
    #crear una columna para extraer la semana del año
    datos['date']=datos.index
    #aplicar una función para extraer la semana del año. Los valores van del 1 al 51
    datos['week'] = datos['date'].apply(lambda x: x.isocalendar()[1])
    #borrar la columna creada
    del datos['date']
    
    return datos
    
```


```python
# Analizar los vacios de las series de datos
def analisis_vacios(datos,rango_analisis,todas_estaciones):
    inicio = rango_analisis[0]
    final = rango_analisis[1]
    # Crear un DataFrame para revisar
    resumen = pd.DataFrame()
    #crear una matriz para almacenar el número de vacios
    por_vacios = pd.Series()
    len_datos = pd.Series()
    num_vacios = pd.Series()

    # isnull(matriz) = permite identificar los valores nulos en una matriz
    # matriz.to_numpy.nonzero()[0] = obtener un array de los valores no tienen datos
    # matriz.loc[filas,columnas] = permite extraer una matriz de datos [filas,columnas]
    # len(matriz) = longitud de una array
    for i in todas_estaciones:
        num_vac = len(pd.isnull(datos.loc[inicio:final,i]).to_numpy().nonzero()[0])
        por_vac = round((100*num_vac/len(datos[i])),2)
        num_vacios[i] = num_vac
        len_datos[i] = len(datos.loc[inicio:final,i])
        por_vacios[i] = por_vac


    # num_vacios['P120'] = len(pd.isnull(datosP120.loc[:,'VALOR']).to_numpy().nonzero()[0])
    resumen['Porcentaje Vacios'] = por_vacios
    resumen['Total de datos'] = len_datos
    resumen['Numero vacios'] = num_vacios
    print(resumen)

```


```python
def llenar_serie(datos,rango_analisis,indices):
    inicio = rango_analisis[0]
    final = rango_analisis[1]
    # Crear un dataframe para los datos a procesar
    datos_procesar = pd.DataFrame()
    # datos que van a entrar en la red neuronal
    datos_procesar = datos.loc[inicio:final,:].copy()
    
    # completar los datos de las estaciones con los valores promedio
    for i in indices:
        datos_procesar[i] = datos_procesar[i].fillna(datos_procesar[i].mean())
    return datos_procesar
    
    
```


```python
#### Entrenar el modelo para obtener los datos de predición
def entrenar_modelo(datos,datos_procesar,estaciones_train,rango_analisis):
    inicio = rango_analisis[0]
    final = rango_analisis[1]
    
    #### Selección de datos para ingresar en el proceso
    # est1 y est2 son las estaciones mas completas
    est1 = estaciones_train[0]
    est2 = estaciones_train[1]
    # est3 estación a completar con la red neuronal
    est3 = estaciones_train[2]
    
    #### Asignar valores a las variables de entrenamiento
    
    # En la variable X_train van los datos de la estaciones más completas y el valor de semana
    X_train = datos_procesar.loc[inicio:final,[est1,est2,'week']].astype(np.float32).values
    # En la varianle y_train va los datos de la estación a rellenar.
    y_train = datos_procesar.loc[inicio:final,est3].astype(np.float32).values

    
    #obtener el valor maximo de la serie
    
    maximo = y_train.max()
    minimo = y_train.min()
    print("Valor Máximo Datos Originales",maximo)
    print("Valor Mínimo Datos Originales",minimo)

    #### Estandarizar la serie de datos eliminando la media y escalando a la varianza de la unidad
    
    # Definir el escalar 
    #scaler = StandardScaler().fit(X_train)
    # Cambiar la serie transformando la serie a valores entre -1 a 1
    #X_train = scaler.transform(X_train)
    
    X_train = get_escalar(X_train)
    
    #### Entrenamiento del modelo
    model = Sequential()
    
    model.add(Dense(12, activation='relu', input_shape=(3,),kernel_constraint=min_max_norm(min_value=minimo, max_value=maximo)))
    model.add(Dense(8, activation='linear'))
    model.add(Dense(1, activation='linear'))
    
    #model.summary()
    
    
    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
                   
    model.fit(X_train, y_train,epochs=300,verbose=0)
    
    # cargar en una variable los datos que arroja la red reuronal
    y_pred = model.predict(X_train)
    
    maximo = y_pred.max()
    minimo = y_pred.min()
    print("Valor Máximo Datos Simulación",maximo)
    print("Valor Mínimo Datos Simulación",minimo)
    
    # calcular la correlacion entre los datos originales y simulados
    datos_correlacion = pd.DataFrame()
    datos_correlacion['Original'] = datos.loc[inicio:final,est3].copy()
    datos_correlacion['Simulado'] = y_pred
    corr = datos_correlacion.corr()
    print("Valor de Correlacion",corr.at['Original', 'Simulado'])
    
    #graficar los datos de predicción
    plot(datos_procesar.index,y_pred,label='Simulado')
    #graficar los datos de est3
    datos.loc[inicio:final,est3].plot(figsize=(16, 4),label='Original'); plt.legend(loc='best')    
        
    return model
```


```python
def get_datos_simulados(model,datos,estaciones_train,rango_analisis):
    inicio = rango_analisis[0]
    final = rango_analisis[1]
    
    # matriz para los datos simulados
    datos_simulados = pd.Series()  
    
    #### Selección de datos para ingresar en el proceso
    # est1 y est2 son las estaciones mas completas
    est1 = estaciones_train[0]
    est2 = estaciones_train[1]
    # est3 estación a completar con la red neuronal
    est3 = estaciones_train[2]
    # Obtener el rango de datos que esta con vacios
    rango = pd.isnull(datos.loc[inicio:final,est3]).to_numpy().nonzero()[0]
    if len(rango)>0:
        # Obtener los valores de est1, est2 basado en el rango de vacios de est3
        X_missing = datos.loc[inicio:final,[est1,est2,'week']].iloc[rango].astype(np.float32).values

        X_missing = get_escalar(X_missing)

        #X_missing[:10]

        # obtener los valores simulados con base al entrenamiento del modelo
        num_datos = len (X_missing)
        y_missing = model.predict(X_missing)
        y_missing = y_missing.reshape([num_datos]).tolist()

        datos_simulados = datos.loc[inicio:final,est3].copy()
        # agregar los datos simulados a la estación P043
        datos_simulados.iloc[rango]=y_missing
    
    return datos_simulados
    
```


```python
def get_escalar(matriz):
    escalar = pd.DataFrame()
    # Definir el escalar 
    scaler = StandardScaler().fit(matriz)
    # Cambiar la serie transformando la serie a valores entre -1 a 1
    escalar = scaler.transform(matriz)
    
    return escalar
    
    
```


```python
def calcular_correlacion(datos):
    # calcular la correlación de los datos
    corr = datos.corr()
    #corr = datos.corr()
    # generar la maskara para el grafico
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    # relación de aspecto del grafico
    plt.figure(figsize=(7,5))
    #grafico de correlación
    sns.heatmap(corr, annot=True)
    #Apply xticks
    plt.xticks(range(len(corr.columns)+1), corr.columns);
    #Apply yticks
    plt.yticks(range(len(corr.columns)+1), corr.columns)
    #show plot
    plt.show()

```


```python

```
