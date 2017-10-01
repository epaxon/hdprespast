import numpy as np
import scipy
import pylab
import cv2
import PIL
from PIL import Image


import matplotlib.pyplot as plt
from align_image_code import align_images
from scipy.ndimage import gaussian_filter
from scipy import misc
from scipy import ndimage

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if '.DS_Store' not in filename and '.ds_store' not in filename:
            img = skio.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
    return images

"""
Part 2: Gradient Domain Fushion
Part 2.1 Toy Problem (10 pts)
"""
im1 = Image.open('data/toy_problem.png')
im1 = np.array(im1, dtype=float)

horiz_grad = np.gradient(im1, axis=0)
vert_grad = np.gradient(im1, axis=1)
orig_pixel = [0,0]
print orig_pixel

plt.figure(figsize=(5,5))
plt.imshow(PIL.Image.fromarray(horiz_grad))
plt.show

plt.figure(figsize=(5,5))
plt.imshow(PIL.Image.fromarray(vert_grad))
plt.show

[imh, imw] = im1.shape
im2var = np.zeros((imh, imw))

val = 0
for i in range(im2var.shape[0]):
    for j in range(im2var.shape[1]):
        im2var[i,j] = val
        val += 1


# sparse matrix
A = np.zeros((2*im1.shape[0]*im1.shape[1]+1, 2*im1.shape[0]*im1.shape[1]+1))
for i in range(A.shape[0]):
    A[i,i] = 1
    if i+1 < A.shape[0]:
        A[i,i+1] = -1
        
b = np.zeros(2*im1.shape[0]*im1.shape[1] + 1)
b[:im1.shape[0]*im1.shape[1]] = np.ravel(horiz_grad)
b[im1.shape[0]*im1.shape[1]:b.shape[0]-1] = np.ravel(vert_grad)
b[b.shape[0]-1] = im1[0,0]

v = scipy.sparse.linalg.lsqr(A, b)[0]
print (A.shape, b.shape)

corner = v[v.shape[0]-1]

img = np.reshape(v[:v.shape[0]-1], (imh, imw))
scipy.misc.imsave('outfile.jpg', img)

