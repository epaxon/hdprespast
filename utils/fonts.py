# General util functions for MSC
import string
import skimage.io as skio
import skimage as sk
import numpy as np
import scipy.ndimage
from skimage import feature

alpha = string.ascii_lowercase

def EditFont(dir="data/font/roman2/"):
  for char in alpha:
    im = skio.imread(dir+"alphabet-letter-"+char+".jpg")
    im[.8*im.shape[0]:,:] = np.max(im)
    skio.imsave("data/font/roman2edit/"+char+".jpg", im)


def LoadFont(dir="data/font/roman2/"):
  l = []

  for char in alpha:
    im = skio.imread(dir+char+".jpg")

    im[im < np.mean(im)] = np.min(im)
    im[im > np.mean(im)] = np.max(im)

    l.append(feature.canny(im))
  s = np.array(l)
  return s

def RotateImage(image, angle):
  if angle < 0:
    angle = 360+angle
  if angle >= 90:
    image = np.rot90(image, angle//90)
    angle = angle-(angle//90)*90
  if angle == 0:
    return image
  x = scipy.ndimage.interpolation.rotate(image, angle, cval=np.min(image))
  l = x.shape[0]-1
  original = image.shape[0]
  middle = l-image.shape[0]
  im = x[middle/2:middle/2+original, middle/2:middle/2+original]
  return im
  