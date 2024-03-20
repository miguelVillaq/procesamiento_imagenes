import numpy as np
import nibabel as nib
# import matplotlib.pyplot as plt

def threshold(img, umbral):
    img_umbral = img > umbral 
    return img_umbral.astype(np.uint8)

def isodata(img, umbral, tolerancia):
    while True:
        img_umbral = threshold(img, umbral)
        img_foreground = img[img_umbral == 1].mean()
        img_background = img[img_umbral == 0].mean()
        umbral_new = (img_foreground + img_background)/2
        if abs(umbral_new - umbral) < tolerancia:
            return img_umbral.astype(np.uint8)
        else:
            umbral = umbral_new
            
def region_growing2D(img, tol, row, col, dep, img_bin, cluster, visited):
  alto, ancho = img.shape[:2]
  stack = [tuple([row, col, dep])]
  while stack:
    valor_comparacion = cluster.mean()
    # Recorrido 8 vecinos.
    for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,1), (1,1), (1,-1), (-1,-1)]:
      next_row, next_col = row + x, col + y
      # Validaciones de que esté dentro de los límites imágenes y que no se haya revisado con anterioridad.
      if 0 <= next_row < alto and 0 <= next_col < ancho:
        if not (tuple([next_row, next_col, dep]) in visited):
          # Comparación del valor de referencia, mete al stack, une al cluster, etc.
          if abs( np.linalg.norm(valor_comparacion - img[next_row, next_col, dep])) <= tol:
            cluster = np.append(cluster, [img[next_row, next_col, dep]], axis = 0)
            img_bin[next_row, next_col, dep] = 1
            stack.append([next_row, next_col, dep])
            visited.add(tuple([next_row, next_col, dep]))
          else:
            #stack.append([next_row, next_col, dep])
            visited.add(tuple([next_row, next_col, dep]))
    row, col, dep = stack.pop(0)
  return img_bin, cluster, visited

def region_growing3D(img, tol, row, col, dep, iter):
  # Parámetros iniciales.
  profundidad = img.shape[2]
  valor_comparacion = img[row,col,dep]
  img_bin = np.zeros_like(img, dtype=np.int8)
  img_bin[row,col,dep] = 1
  cluster = np.array([valor_comparacion])
  visited = set()
  stack_dep = []

  # Ejecución inicial del algoritmo 2D, es decir, una sola profundidad.
  img_ini, cluster, visited = region_growing2D(img, tol, row, col, dep, img_bin, cluster, visited)

  # Ciclo que maneja cuántas iteraciones de profundidad se harán.
  for i in range(iter):
    for z in [(1),(-1)]:
      next_dep = dep + z
      if 0 <= next_dep < profundidad:
        if not (tuple([row, col, next_dep]) in visited):
          img_ini, cluster, visited = region_growing2D(img, tol, row, col, next_dep, img_ini, cluster, visited)
          stack_dep.append(next_dep)
    dep = stack_dep.pop(0)
  return img_ini.astype(np.uint8)

def kmeans(img, k, num_iter):
    # Crear los centroides iniciales aleatorios
    cluster = np.random.choice(np.ravel(img[img > 0]), size=k, replace=False)
    #print(cluster)
    #cluster = np.linspace(np.amin(img), np.amax(img), k)

    for _ in range(num_iter):  # Realizamos 10 iteraciones (puedes ajustar este valor según sea necesario)
        # Calcular las distancias entre cada píxel y los centroides, con expand_dims, reducimos trabajo al calcular la resta sólo en la dimensión de profundidad.
        distance = np.abs(np.expand_dims(img, axis=-1) - cluster)

        # Asignar cada píxel al cluster más cercano: Segmentation contiene los índices del cluster al que pertenece la distancia más corta.
        segmentation = np.argmin(distance, axis=-1)

        # Actualizar los centroides como el promedio de los píxeles asignados a cada cluster
        for k_cluster in range(k):
            cluster[k_cluster] = np.mean(img[segmentation == k_cluster])

    # Calcular la segmentación final
    segmentation = np.argmin(np.abs(np.expand_dims(img, axis=-1) - cluster), axis=-1)
    return segmentation.astype(np.uint8)