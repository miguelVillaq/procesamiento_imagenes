import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt



def denoising(img,row,col,dep,value,shape,tol):
  stack = np.array([value])
  alto, ancho = shape[:2]
  for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,1), (1,1), (1,-1), (-1,-1)]:
    next_row, next_col = row + x, col + y
    if 0 <= next_row < alto and 0 <= next_col < ancho:
      if abs(img[row, col, dep] - img[next_row,   next_col, dep]) < tol:
        # Meter en donde se sacará el promedio.
        stack = np.append(stack, img[next_row, next_col, dep])

  return stack

def denoising_img(img, dep, tol, reps, exec):
  img_bin = np.zeros_like(img)
  shape = img.shape
  for row in range(shape[0]):
    for col in range(shape[1]):
      value = img[row, col, dep]
      denoising_around = denoising(img, row, col, dep, value, shape, tol)
      desnoising_dep = denoising_dep(img, reps, row, col, dep, shape[2], tol)
      denoising_total = exec(np.concatenate((denoising_around, desnoising_dep)))
      img_bin[row,col,dep] = denoising_total
  return img_bin


def denoising_dep(img, reps, row, col, z_dep, size_z, tol):
  stack = []
  result = []
  shape = img.shape
  for iter in range(reps):
    for z in[(1),(-1)]:
      next_dep = z_dep + z
      if 0 <= next_dep < size_z:
        value = img[row,col,next_dep]
        result = np.concatenate((result, denoising(img, row, col, next_dep, value, shape, tol)))
        stack.append(next_dep)
    z_dep = stack.pop(0)
  return result

def mean(array):
  return array.mean()

def median(array):
  return np.median(array)

# img = nib.load("img\sub-01_T1w.nii")
# img = img.get_fdata()

# result1 = denoising_img(img, 121, 50, 2, median)
# result2 = denoising_img(img, 121, 50, 2, mean)


# plt.figure(figsize=(16,9))
# plt.subplot(2,3,1)
# plt.title('Original')
# plt.imshow(img[:,:,121])#,cmap='gray')
# plt.subplot(2,3,2)
# plt.title('Denoising median 1')
# plt.imshow(result1[:,:,121])#,cmap='gray')
# plt.subplot(2,3,3)
# plt.title('Denoising mean 1')
# plt.imshow(result2[:,:,121])#,cmap='gray')
# plt.show()