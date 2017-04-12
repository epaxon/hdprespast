# General util functions for MSC
import string
import skimage.io as skio
import skimage as sk
import numpy as np
import scipy.ndimage
from skimage import feature
from scipy import misc

alpha = string.ascii_lowercase

def EditFont(dir="data/font/64/"):
  for char in alpha:
    im = skio.imread(dir+"alphabet-letter-"+char+".jpg")
    im[.8*im.shape[0]:,:] = np.max(im)
    skio.imsave("data/font/roman2edit/"+char+".jpg", im)


def LoadFont(dir="data/font/64/"):
  l = []

  for char in alpha:
    im = skio.imread(dir+char+".jpg")
    im = 255-im
    im = feature.canny(im, sigma=1)
    z = misc.imresize(im, .5)
    im = np.zeros(im.shape)
    length = z.shape[0]
    x = im.shape[0]//2-length//2
    im[x:x + length, x:x + length] = z


    l.append(im)
  # s = np.array(l)
  return l

def RotateImage(image, angle):
  cval = image[0][0]
  im = misc.imrotate(image, angle)
  return im
  