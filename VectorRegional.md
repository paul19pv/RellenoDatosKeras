

```python
from sklearn import linear_model
from numpy import genfromtxt
%pylab inline
import pandas as pd
import numpy as np
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
def generar_correlacion(datos):
    ### generamos la correlacion lineal con LinearRegresion 
    ecuacion=[]
    clf = linear_model.LinearRegression()
    clf.fit(datos.iloc[:,:3],datos.iloc[:,3:])
    
    coeficientes = clf.coef_
    #ecuacion.append(clf.intercept__)
    factor = clf.intercept_
    ecuacion = np.append(coeficientes, factor)
    ecuacion = list(ecuacion)
    # print(list(ecuacion))
    return ecuacion

```


```python
# leer el archivo con la información de la estaciones
archivo = pd.ExcelFile('datos_rellenados.xlsx')
datos = pd.read_excel(io=archivo,index_col=0,parsedates=True,sheet_name='Sheet1')
datos.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>P120</th>
      <th>P364</th>
      <th>P349</th>
      <th>P043</th>
      <th>P121</th>
      <th>P532</th>
      <th>P622</th>
    </tr>
    <tr>
      <th>FECHA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1963-01-01</th>
      <td>98.599998</td>
      <td>249.000000</td>
      <td>204.982086</td>
      <td>136.626236</td>
      <td>100.176987</td>
      <td>102.138268</td>
      <td>105.561844</td>
    </tr>
    <tr>
      <th>1963-02-01</th>
      <td>130.199997</td>
      <td>159.699997</td>
      <td>181.016479</td>
      <td>153.961151</td>
      <td>71.781181</td>
      <td>66.388359</td>
      <td>88.614380</td>
    </tr>
    <tr>
      <th>1963-03-01</th>
      <td>154.300003</td>
      <td>188.899994</td>
      <td>201.716873</td>
      <td>140.271667</td>
      <td>81.764816</td>
      <td>60.359131</td>
      <td>86.530258</td>
    </tr>
    <tr>
      <th>1963-04-01</th>
      <td>110.000000</td>
      <td>111.099998</td>
      <td>125.213799</td>
      <td>100.379921</td>
      <td>65.364723</td>
      <td>34.400215</td>
      <td>54.454197</td>
    </tr>
    <tr>
      <th>1963-05-01</th>
      <td>62.900002</td>
      <td>80.900002</td>
      <td>78.963104</td>
      <td>66.699997</td>
      <td>59.084435</td>
      <td>36.915268</td>
      <td>49.523899</td>
    </tr>
  </tbody>
</table>
</div>




```python
# leer el archivo con la ubicación y altura de la estaciones
archivo_est = pd.ExcelFile('HA_Stations.xlsx')
ubicacion = pd.read_excel(io=archivo_est,index_col=0,sheet_name='Stations')
ubicacion.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nombre</th>
      <th>Latitud</th>
      <th>Longitud</th>
      <th>Altitud</th>
    </tr>
    <tr>
      <th>Codigo</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>P043</th>
      <td>MARISCAL SUCRE INAMHI</td>
      <td>-0.637500</td>
      <td>-78.490829</td>
      <td>3670</td>
    </tr>
    <tr>
      <th>P120</th>
      <td>COTOPAXI-CLIRSEN</td>
      <td>-0.618167</td>
      <td>-78.569832</td>
      <td>3560</td>
    </tr>
    <tr>
      <th>P121</th>
      <td>EL REFUGIO-COTOPAXI</td>
      <td>-0.655500</td>
      <td>-78.568657</td>
      <td>4800</td>
    </tr>
    <tr>
      <th>P349</th>
      <td>HDA.PINANTURA(LA COCHA)</td>
      <td>-0.421667</td>
      <td>-78.353333</td>
      <td>3250</td>
    </tr>
    <tr>
      <th>P364</th>
      <td>LORETO PEDREGAL</td>
      <td>-0.556833</td>
      <td>-78.422501</td>
      <td>3620</td>
    </tr>
  </tbody>
</table>
</div>




```python
datos['date']=datos.index
datos['month'] = datos['date'].apply(lambda x: x.month)
del datos['date']
# datos.head()
promedios = datos.groupby('month').mean()
promedios
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>P120</th>
      <th>P364</th>
      <th>P349</th>
      <th>P043</th>
      <th>P121</th>
      <th>P532</th>
      <th>P622</th>
    </tr>
    <tr>
      <th>month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>100.864006</td>
      <td>158.195316</td>
      <td>145.556872</td>
      <td>143.809767</td>
      <td>90.796256</td>
      <td>87.667688</td>
      <td>127.187843</td>
    </tr>
    <tr>
      <th>2</th>
      <td>116.095743</td>
      <td>158.597727</td>
      <td>165.432818</td>
      <td>142.858079</td>
      <td>81.059008</td>
      <td>81.645428</td>
      <td>107.020569</td>
    </tr>
    <tr>
      <th>3</th>
      <td>142.701626</td>
      <td>192.686363</td>
      <td>192.562789</td>
      <td>147.911956</td>
      <td>97.379102</td>
      <td>84.306188</td>
      <td>109.923278</td>
    </tr>
    <tr>
      <th>4</th>
      <td>140.435801</td>
      <td>163.872727</td>
      <td>189.526465</td>
      <td>142.375111</td>
      <td>87.478111</td>
      <td>68.314256</td>
      <td>93.809758</td>
    </tr>
    <tr>
      <th>5</th>
      <td>115.901431</td>
      <td>128.125000</td>
      <td>135.037360</td>
      <td>118.254878</td>
      <td>71.565893</td>
      <td>47.797554</td>
      <td>60.709088</td>
    </tr>
    <tr>
      <th>6</th>
      <td>68.505489</td>
      <td>63.313637</td>
      <td>79.012148</td>
      <td>87.294441</td>
      <td>52.477340</td>
      <td>50.850009</td>
      <td>47.846255</td>
    </tr>
    <tr>
      <th>7</th>
      <td>47.633625</td>
      <td>38.022727</td>
      <td>60.881888</td>
      <td>77.054898</td>
      <td>50.859932</td>
      <td>56.481857</td>
      <td>40.608785</td>
    </tr>
    <tr>
      <th>8</th>
      <td>40.399564</td>
      <td>36.531818</td>
      <td>64.462337</td>
      <td>72.088903</td>
      <td>51.290258</td>
      <td>52.360080</td>
      <td>31.195027</td>
    </tr>
    <tr>
      <th>9</th>
      <td>80.329837</td>
      <td>91.631818</td>
      <td>120.070465</td>
      <td>81.768063</td>
      <td>66.714411</td>
      <td>47.781374</td>
      <td>45.880593</td>
    </tr>
    <tr>
      <th>10</th>
      <td>114.092705</td>
      <td>158.150000</td>
      <td>166.182979</td>
      <td>125.693324</td>
      <td>93.797357</td>
      <td>79.518107</td>
      <td>105.797630</td>
    </tr>
    <tr>
      <th>11</th>
      <td>104.160344</td>
      <td>168.336363</td>
      <td>163.533006</td>
      <td>127.332369</td>
      <td>113.234521</td>
      <td>78.648278</td>
      <td>116.449369</td>
    </tr>
    <tr>
      <th>12</th>
      <td>105.100820</td>
      <td>162.863635</td>
      <td>154.063753</td>
      <td>134.797686</td>
      <td>103.163665</td>
      <td>79.337102</td>
      <td>127.877577</td>
    </tr>
  </tbody>
</table>
</div>




```python
#### Unir los datos de promedios y la informacion de Latitud, Longitud y Altitud
datos_procesar = ubicacion.loc[:,['Latitud','Longitud','Altitud']].copy()
todas_estaciones = ['P120','P364','P349','P043','P121','P532','P622']
for mes in range(1,13):
    datos_procesar[mes]=promedios.loc[mes,todas_estaciones]

```


```python
# Correlacion
matriz_ecuaciones = []

for item in range(3,15):
    datos_corr = datos_procesar.iloc[:,[0,1,2,item]].copy()
    ecuacion = generar_correlacion(datos_corr)
    ecuacion.insert(0,item-2)
    matriz_ecuaciones.append(ecuacion)



ecuaciones = pd.DataFrame(matriz_ecuaciones, columns = ['Mes', 'Coef_Latitud','Coef_Longitud','Coef_Altitud','FactorIndependiente']) 
#print(matriz_ecuaciones)

#generar_correlacion(datos_corr)
ecuaciones.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mes</th>
      <th>Coef_Latitud</th>
      <th>Coef_Longitud</th>
      <th>Coef_Altitud</th>
      <th>FactorIndependiente</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-357.644887</td>
      <td>438.276784</td>
      <td>-0.022168</td>
      <td>34394.288593</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-312.241964</td>
      <td>323.985257</td>
      <td>-0.042787</td>
      <td>25528.586832</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-281.478319</td>
      <td>268.545010</td>
      <td>-0.048613</td>
      <td>21233.808068</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-185.512870</td>
      <td>113.227508</td>
      <td>-0.055561</td>
      <td>9114.754433</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-266.784603</td>
      <td>97.023229</td>
      <td>-0.049433</td>
      <td>7745.540606</td>
    </tr>
  </tbody>
</table>
</div>




```python
# exportar los datos a excel
ecuaciones.to_excel('ecuaciones.xlsx')
```


```python

```
