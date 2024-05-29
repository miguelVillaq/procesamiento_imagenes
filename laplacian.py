import numpy as np
from scipy import ndimage
from scipy.sparse import lil_matrix, find, diags, csr_matrix
import scipy.sparse.linalg as spla
import math

# Matriz de adyacencia/pesos -> Ecuación #1 y 3.2.
def calcular_peso(Int_i, Int_j, beta, sigma):
    peso = ((beta * np.power(np.abs(Int_i - Int_j), 2))/sigma) * -1
    return np.exp(peso)

def grafo_pesos(img, beta):
    # 8 vecinos.
    vecinos = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,1), (1,1), (1,-1), (-1,-1)]
    # dimensiones
    alto, ancho = img.shape
    # Cantidad voxeles.
    vox = alto*ancho
    m_adyacencia = lil_matrix((vox,vox))
    
    # Recorriendo la img.
    for i in range(alto):
        for j in range(ancho):
            # Índice líneal de posición (nodo) actual.
            i_nodo = i * ancho + j
            # Revisando vecinos.
            for x,y in vecinos:
                new_i, new_j = i+x, j+y
                
                # Coordenadas dentro de la img.
                if 0 <= new_i < alto and 0 <= new_j < ancho:
                    # Índice líneal vecino.
                    i_vecino = new_i * ancho + new_j
                    
                    sigma = np.max(np.abs(img[i,j] - img[new_i,new_j])) + np.power(10.0,-6)
                    # Peso arista.
                    peso = calcular_peso(img[i,j], img[new_i,new_j], beta, sigma)
                    peso = np.power(10.0,-6) if peso == 0 else peso
                    m_adyacencia [i_nodo,i_vecino] = peso
    m_adyacencia = m_adyacencia.tocsr()
    return m_adyacencia

def suma_pesos_vox(m_ady, img):
    alto, ancho = img.shape
    img2d = np.zeros(img.shape)
    fil, col, vlr = find(m_ady)
    #Recorrer aristas m_adyacencia.
    for fila,valor in zip(fil,vlr):
        # Coordenadas 2D del nodo.
        i = fila // ancho
        j = fila % ancho
        img2d[i,j] +=valor
    return img2d

# Ecuación 7.
def solv_sistema_lineal(sum_pesos, m_ady, img, back, foreg):
    sum_pesos_flat = sum_pesos.flatten()
    
    L = diags(sum_pesos_flat) - m_ady
    L += np.power(10.0,-6) * diags(np.ones(len(sum_pesos_flat))) 
    
    Is = np.zeros_like(img)
    b = np.zeros_like(img)
    
    for semilla in back:
        Is[(semilla)] = 1
        b[(semilla)] = img[(semilla)]
    for semilla in foreg:
        Is[(semilla)] = 1
        b[(semilla)] = img[(semilla)]
        
    b_flat = b.flatten()
    Is_flat = Is.flatten()
    
    Is_diag = diags(Is_flat)
    Ex = Is_diag + np.power(L,2)
    
    # Halla valores de x.
    min_sparse = csr_matrix(x)
    x = spla.cg(min_sparse,b_flat)
    
    return x[0]

# Ecuación 3.
def etiquetado_final(img,sol_x, back, foreg):
    new_img = np.zeros_like(img)
    
    vlr_back = np.mean(img[back[:, 0], back[:, 1]])
    vlr_foreg = np.mean(img[foreg[:, 0], foreg[:, 1]])
    
    prom = (vlr_back + vlr_foreg) / 2
    
    for i, xi in enumerate(sol_x):
        coordenada = np.unravel_index(i,img.shape)
        if xi >= prom:
            new_img[coordenada] = 1 
        else:
            new_img[coordenada] = 0 
            
    return new_img

def ejecutar(img, beta, back, foreg):
    m_ady = grafo_pesos(img, beta)
    #print("A: ",m_ady)
    pesos_aristas = suma_pesos_vox(m_ady, img)
    #print("B: ",pesos_aristas )
    x_result = solv_sistema_lineal(pesos_aristas, m_ady, img, back, foreg)
    #print("C: ",x_result)
    new_img = etiquetado_final(img, x_result, back, foreg)
    return new_img