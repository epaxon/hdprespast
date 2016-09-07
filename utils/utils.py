import numpy as np

def normalize(image1):
  return (image1-np.mean(image1))/np.std(image1)

def NCC(image1, image2):
  return np.mean(normalize(image1) * normalize(image2))


def GenerateRandomBinary(n, d):
  x = np.random.randn(n, d)
  x[x<0] = -1
  x[x>0] = 1
  return x
