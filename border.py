import numpy as np
import scipy.ndimage as sn

def derivada_primer_orden(img, punto, eje, umbral):
    kernel_x = np.array([[0,0,0],[-1,0,1],[0,0,0]])/2
    kernel_y = np.array([[0,-1,0],[0,0,0],[0,1,0]])/2
    seccion = []
    if (eje == "x"):
        seccion = img[punto,:,:]
    if (eje == "y"):
        seccion = img[:,punto,:]
    if (eje == "z"):
        seccion = img[:,:,punto]
    convulucion_x = sn.convolve(seccion,kernel_x) 
    convulucion_y = sn.convolve(seccion,kernel_y) 
    magnitud_convolucion = np.sqrt(convulucion_x ** 2 + convulucion_y ** 2) > umbral #50
    seccion = magnitud_convolucion
    return seccion

#resultado = derivada_primer_orden(img,120,"z",50)
#plt.imshow(resultado)

def derivada_segundo_orden(img, punto, eje, umbral):
    kernel_x = np.array([[0,0,0],[-1,0,1],[0,0,0]])/2
    kernel_y = np.array([[0,-1,0],[0,0,0],[0,1,0]])/2
    seccion = []
    if (eje == "x"):
        seccion = img[punto,:,:]
    if (eje == "y"):
        seccion = img[:,punto,:]
    if (eje == "z"):
        seccion = img[:,:,punto]
    convulucion_x = sn.convolve(seccion,kernel_x) 
    convulucion_xx = sn.convolve(convulucion_x,kernel_x) 
    convulucion_y = sn.convolve(seccion,kernel_y)
    convulucion_yy = sn.convolve(convulucion_y,kernel_y)  
    magnitud_convolucion = np.sqrt(convulucion_xx ** 2 + convulucion_yy ** 2) > umbral #40
    seccion = magnitud_convolucion
    return seccion

def dif_filtro(img,punto,eje,umbral):
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/9
    seccion = []
    if (eje == "x"):
        seccion = img[punto,:,:]
    if (eje == "y"):
        seccion = img[:,punto,:]
    if (eje == "z"):
        seccion = img[:,:,punto]
    convolucion = sn.convolve(seccion, kernel)
    diferencia = (seccion-convolucion) > umbral #23
    seccion = diferencia
    return seccion
    