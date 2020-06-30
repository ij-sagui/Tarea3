# Tarea3
Punto 1:
A partir de las gráficas obtenidas de los vectores de función de densidad marginal de los datos, se observa que la curva de mejor a ajuste para los datos corresponde a la de densidad gausssiana, tanto para X como para Y.

Punto2:
Asumiendo la independencia estadística de las variable X y Y, la función de desidad conjunta corresponde a la multiplicación de las funciones de densidad marginales, es decir, fxy(x,y)=fx(x)fy(y). Esta se muestra en el punto 2 en el código de python.

Punto 3:
A partir del código utilizado se tiene que la correlación es Rxy=149,5428 y de los valores esperados también calculados se tiene que Rxy se puede expresar como Rxy=E[X]E[Y]. Esto último quiere decir las variables aleatorias no están correlacionadas, lo que quiere decir que no hay asociación lineal entre los datos de X y los datos de Y.
La covarianza obtenido es Cxy=0.058761. La covarianza indica el grado de variación conjunta entre las varaibles aleatorias, como ejemplo, si hay valores positivos de covarianza indicaría que si la variable X aumenta Y tambíen aumenta. Para en este caso la covarianza es muy cercana a 0, lo que quiere decir que hay un bajo grado de variación conjunta entre las variables aleatorias.
El coeficiente de Pearson obtenido es p=0,003949998 e indica que existe una leve correlación positiva entre X y Y si no se toma como cero. Si se toma como cero indica que no hay asociación lineal entre las variables.

Punto 4:
Las gráficas fmarginalx y fmarginaly muestran en comportamiento de las funciones de desidad marginales calculadas a partir de los datos. Las gráficas mejorajustefx y mejor ajustefy muestran las gráficas de las funciones de densidad de mejor ajuste para los datos obtenidos y se encuentran superpuestas a las gráficas de densidad obtenidad a partir de los datos. Por último, la imagen Densidadconjunta muestra la función de densidad conjunta calculada a partir de las curvas de mejor ajuste obtenidas y tomando en cuenta la independencia estadísticas de X y Y.
