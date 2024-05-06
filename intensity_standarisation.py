import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.stats as sc
from collections import Counter


# Intensidad
# Intensity rescaling

def intensity_rescaling(img):
  min = np.min(img)
  max = np.max(img)
  new_img = (img - min)/(max-min)
  return new_img

#img_intensity = intensity_rescaling(img)

#plt.imshow(img_intensity[:,:,120])
# plt.hist(img_intensity[img_intensity > 0.01].flatten(), 100)

# Intensidad
# Z-Score

def z_score(img):
  mean = img[img>10].mean()
  deviation = img[img>10].std()
  new_img = (img - mean)/deviation
  return new_img
#img_z_score = z_score(img)

#plt.imshow(img_z_score[:,:,120])
# plt.hist(img_z_score.flatten(), 50)

# Intensidad
# Histograma

# El algoritmo funciona, pero se tarda 5-10 en completarse, adicionalemente requiere 1GB de ram libre, o cercano.
# Necesita mínimo 3 puntos para funcionar.
def n_matching(trainData, testData, k):
  #trainData = np.array([[1, 2, 3], [4, 5, 6],[7,8,9]])
  function,landmark = training(trainData,k)
  test = testing(landmark,testData, function,k)
  return test

def aux(m,b):
  return lambda x: (m*x) + b

def training(inputData, k):
  # Array contenedor de función a trozos.
  functions = []
  # Marcas de percentiles: K percentiles equidistantes.
  x = np.linspace(5, 95, k)
  # Saca los percentiles de X de la imágen.
  y = np.percentile(inputData.flatten(), x)
  for i in range(1,k):
    m = (y[i]-y[i-1])/(x[i]-x[i-1])
    # y = mx + b -> Despeja b
    b = y[i-1] - m * x[i-1]
    functions.append(aux(m,b))
  return functions, y

# def testing(landmark, inputData, function):
#   new_img = np.zeros_like(inputData)
#   intensities = sorted(list(set(inputData.flatten())), reverse=False)
#   # Como no cabe en ram todo el proceso de percentiles, entonces se hace por cada slide de X.
#   for x_slice in range(inputData.shape[0]):
#     # Se convierten todos las intensidades en percentiles.
#     #percentil = sc.percentileofscore(intensities, inputData)
#     percentil = sc.percentileofscore(intensities, inputData[x_slice])
#     # Se selecciona qué función se usará dado la condición de que esté dentro del otro percentil.
#     for i in range(1,len(landmark)):
#       condicion = (percentil > landmark[i-1]) & (percentil < landmark[i])
#       #new_img[condicion] = function[i-1](percentil[condicion])
#       values = function[i-1](percentil[condicion])
#       for i in range(len(condicion)):
#         for j in range(len(condicion[0])):
#           if condicion[i][j]:
#             new_img[x_slice][i][j],values = values[0], values[1:]
#   return new_img

def testing(landmark, inputData, function, k):
  new_img = np.zeros(inputData.shape)
  intensities = sorted(list(set(inputData.flatten())), reverse=False)
  percentil = sc.percentileofscore(intensities, inputData)
  for i in range(k):
    if(i+1 < k):
      condicion = (percentil > landmark[i]) & (percentil < landmark[i+1])
      new_img[condicion] = function[i](percentil[condicion])
  return new_img

#result = n_matching(img, img, 3)
#data1=np.array([[[10,3,1], [7,6,2], [3,4,1]], [[3,5,2], [2,6,1], [1,2,2]],[[9,3,1],[6,7,2],[5,4,1]]])
#result = n_matching(np.array([[10, 7, 4], [3, 2, 1],[8,6,5]]), np.array([[1, 2, 3], [4, 5, 6],[7,8,9]]), 4)
#result = n_matching(data1, data1, 4)
#print(result)




# plt.figure(figsize=(16,9))
# plt.subplot(2,3,1)
# plt.title('Original')
# plt.imshow(img[:,:,120],cmap='gray')
# plt.subplot(2,3,2)
# plt.title('Original')
# plt.hist(img.flatten(), 50)
# plt.subplot(2,3,3)
# plt.title('histograma')
# plt.imshow(result[:,:,120],cmap='gray')
# plt.subplot(2,3,4)
# plt.title('histograma')
# plt.hist(result.flatten(), 50)

# Intensidad
# White

def white_stripe(img):
  new_img = np.zeros_like(img)
  rangos = np.linspace(5, 95, 5)
  percentiles_img = np.percentile(img.flatten(), rangos)
  intensidades = img[(img < percentiles_img[-1]) & (img > percentiles_img[-2])]
  ws = (Counter(intensidades)).most_common(1)[0][0]
  new_img = img/ws
  return new_img

#result= white_stripe(img)


# plt.figure(figsize=(16,9))
# plt.subplot(2,3,1)
# plt.title('Original')
# plt.imshow(img[:,:,120],cmap='gray')
# plt.subplot(2,3,2)
# plt.title('Original')
# plt.hist(img.flatten(), 50)
# plt.subplot(2,3,3)
# plt.title('white_stripe')
# plt.imshow(result[:,:,120],cmap='gray')
# plt.subplot(2,3,4)
# plt.title('white_stripe')
# plt.hist(result.flatten(), 50)