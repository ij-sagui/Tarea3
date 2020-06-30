# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:36:07 2020

@author: Isaí Saborío Aguilar
"""


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit



#extracción de datos del archivo CSV.
datos=np.array(pd.read_csv('xy.csv',header=0,index_col = 0),dtype=float)

#función  marginal de X.
fx=np.insert(np.sum(datos,axis=1),0,[0]*5)

#función marginal de y
fy=np.insert(np.sum(datos,axis=0),0,[0]*5)

#vectores de las variables aleatorias
x=np.linspace(0,len(fx)-1,len(fx)) #toman en cuenta los puntos anteriores a x5 y y5
y=np.linspace(0,len(fy)-1,len(fy))

#graficación de los datos para seleccionar la curva de mejor ajuste
#gráfica datos de x.
plt.plot(x,fx)
plt.grid('true')
plt.title('Densidad marginal X')
plt.xlabel('X')
plt.ylabel('fx(x)')
plt.savefig('fmarginalx')
#datos de y.
plt.figure()
plt.plot(y,fy)
plt.grid('true')
plt.title('Desidad marginal de Y')
plt.xlabel('Y')
plt.ylabel('f(y)')
plt.savefig('fmarginaly')


'''
Punto 2
Analíticamente f(x,y)=f(x)f(y) debido a la independencia asumida
'''
#de las gráficas se puede observar que las curvas que mejor se ajustan son las de densidad gaussiana. La media parece estar en la parte más alta y son casi simetricas a partir de este punto.
def fXY(x1,y1, mux, sigmax,muy, sigmay):
    return 1/np.sqrt(2*np.pi*sigmax**2)*np.exp(-(x1-mux)**2/(2*sigmax**2))*1/np.sqrt(2*np.pi*sigmay**2)*np.exp(-(y1-muy)**2/(2*sigmay**2))


'''
Punto 3
'''
#cálculo de la correlación
corr=0
for i in range(len(x)-5):
    for k in range(len(y)-5):
       corr+= x[5+i]*y[5+k]*datos[i][k]

#cálculo de medias
meanx=0
meany=0
for l in range(len(y)):
    meany+=y[l]*fy[l]
    if l<16:
        meanx+=x[l]*fx[l]
        
#cálculo coeficiente de correlación
vary=0 
varx=0       
for z in range(len(y)):
    vary+=((y[z]-meany)**2)*fy[z]
    if z<16:
        varx+=((x[z]-meanx)**2)*fx[z]       
        
p=(corr-meanx*meany)/(np.sqrt(vary*varx))        
        
print('La correlación es: Rxy={0}'.format(corr))  
print('El valor esperado de X es: E[X]={0}'.format(meanx))
print('El valor esperado de Y es: E[Y]={0}'.format(meany))
print('La covarianza de X y Y es: Cxy={0}'.format(corr-meanx*meany))
print('El coeficiente de Pearson es: p={0}'.format(p))

    
'''
Punto 4
'''
#Definición de la función de densidad.
def gauss(x2,mu,sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x2-mu)**2/(2*sigma**2))

#parámetros mu y sigma de de los datos aproximandolos a una gaussiana
paramg,_ =curve_fit(gauss, x, fx)
paramg2,_ =curve_fit(gauss, y, fy)

#gráfica de la curva de mejor ajuste para X
plt.figure()
plt.plot(x,fx,label='Datos')
plt.plot(gauss(x,paramg[0],paramg[1]),label='Gaussiana')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid('true')
plt.title('Curva de mejor ajuste para fx')
plt.xlabel('X')
plt.ylabel('fx(x)')
plt.savefig('mejorajustefx')
#gráfica de la curva de mejor ajuste para Y
plt.figure()
plt.plot(y,fy,label='Datos')
plt.plot(gauss(y,paramg2[0],paramg2[1]),label='Gaussiana')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid('true')
plt.title('Curva de mejor ajuste para fy')
plt.xlabel('Y')
plt.ylabel('fy(y)')
plt.savefig('mejorajustefy')
#gráfica de la función de densidad conjunta
fig=plt.figure()
ax=Axes3D(fig)
#Datos para graficar en 3D
datos1=pd.read_csv('xyp.csv',header=0) # datps extraidos del otro archivo
z=fXY(np.array(datos1['x']),np.array(datos1['y']), paramg[0], paramg[1],paramg2[0], paramg2[1])
#Generación de gráfica en 3D
ax.plot_trisurf(np.array(datos1['x']),np.array(datos1['y']),z,cmap=plt.cm.jet)
plt.title('Función de densidad conjunta')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('fxy(x,y)')
plt.savefig('Densidadconjunta')

       