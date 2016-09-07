from utils import fonts, utils
import skimage.io as skio
import numpy as np

roman_alphabet = fonts.LoadFont()

def X_MSC(input_image, template):
  k = np.ones(20)/20.
  while np.max(k) < 2/20.:
    super_imposed = np.zeros(input_image.shape)
    for i in range(20):
      x = np.roll(input_image, i*2, axis=1)
      super_imposed += k[i]*x
      dif = np.sum(np.square(x-template))/(1+10*k[i])
      k[i] = k[i]/(dif+1)
    k = k/np.sum(k)
  return np.argmax(k)*2

def ROT_MSC(input_image, template, k = np.ones(72)/72.):
  """
  Returns the rotation difference between template and input_image
  """
  for i in range(72):
    x = fonts.RotateImage(input_image, i*5)
    dif = np.sum(np.square(x-template))/(1+10*k[i])
    k[i] = k[i]/dif

    k = k/np.sum(k)

  return k

def X_MSC_MSC(input_image, template1, hyper_dim=10000, rot_=10):
  # 1 - all x translations
  # 2 - all y translations
  k1 = np.ones(10)/10.
  k2 = np.ones(10)/10.
  k3 = np.ones(rot_)/float(rot_)
  forward_super_imposed1 = np.zeros(hyper_dim)
  forward_super_imposed2 = np.zeros(hyper_dim)
  r1 = np.zeros((10, hyper_dim))
  r2 = np.zeros((10, hyper_dim))
  r3 = np.zeros((rot_, hyper_dim))

  
  print "l"
  l = input_image.shape[0]*input_image.shape[1]

  k = utils.GenerateRandomBinary(l, hyper_dim)
  deg = 360/rot_

  # fn = utils.NCC
  fn = lambda x, y: np.sum(x*y)

  # Forward prop
  for i in range(10):
    x = np.roll(input_image, (i-5)*8, axis=1)
    x = np.dot(x.reshape(l), k)
    print x.shape
    r1[i] = k1[i]*x
    forward_super_imposed1 += k1[i]*x

  for i in range(10):
    y = np.roll(forward_super_imposed1, (i-5)*8, axis=0)
    r2[i] = k2[i] * y
    forward_super_imposed2 += k2[i]*y

  for i in range(rot_):
    r3[i] = k3[i]*fonts.RotateImage(forward_super_imposed2, i*deg)

  back_super_imposed2 = np.zeros(input_image.shape)
  back_super_imposed3 = np.zeros(input_image.shape)
  # Backward prop
  for i in range(rot_):
    back_super_imposed3 += k3[i] * fonts.RotateImage(template1, i*deg)
    q = fn(template1, r3[i])
    print i, q
    k3[i] = k3[i]*np.exp(q)

  for i in range(10):
    y = np.roll(back_super_imposed3, (i-5)*8, axis=0)
    back_super_imposed2 += k2[i] * y
    q = fn(template1, r2[i])
    k2[i] = k2[i]*np.exp(q)

  for i in range(10):
    q = fn(back_super_imposed2, r1[i])
    k1[i] = k1[i]*np.exp(q)

  k3 = k3/np.sum(k3)
  k2 = k2/np.sum(k2)
  k1 = k1/np.sum(k1)

  print "Rotation: " + str(np.argmax(k3)*deg)
  print "Y-Translate: " + str((np.argmax(k2)-5)*8)
  print "X-Translate: " + str((np.argmax(k1)-5)*8)


rot = -240
ytranslate = 8
xtranslate = 16

im = fonts.RotateImage(roman_alphabet[0], rot)
im = np.roll(im, xtranslate, axis=1)
im = np.roll(im, ytranslate, axis=0)

# rot = MSC(seventyfive, roman_alphabet[0])
sl = X_MSC_MSC(im, roman_alphabet[0], 10000, 24)

    
