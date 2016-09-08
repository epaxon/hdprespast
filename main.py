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

def X_MSC_MSC(input_image, template1, hyper_dim=10000, rot_=10, gran=10, dist=5):
  # 1 - all x translations
  # 2 - all y translations


  k1 = np.ones(gran)/float(gran)
  k2 = np.ones(gran)/float(gran)
  k3 = np.ones(rot_)/float(rot_)
  forward_super_imposed1 = np.zeros(input_image.shape)
  forward_super_imposed2 = np.zeros(input_image.shape)
  r1 = np.zeros((gran, hyper_dim))
  r2 = np.zeros((gran, hyper_dim))
  r3 = np.zeros((rot_, hyper_dim))

  l = input_image.shape[0]*input_image.shape[1]

  deg = 360/rot_

  # fn = utils.NCC
  fn = lambda x, y: np.sum(x*y)

  k1 = np.ones(gran)/float(gran)
  k2 = np.ones(gran)/float(gran)
  k3 = np.ones(rot_)/float(rot_)
  k = utils.GenerateRandomBinary(l, hyper_dim)
  # Forward prop
  for i in range(gran):
    x_im = np.roll(input_image, (i-gran//2)*dist, axis=1)
    x = np.dot(x_im.reshape(l), k)
    r1[i] = x
    forward_super_imposed1 += x_im

  for i in range(gran):
    y_im = np.roll(forward_super_imposed1, (i-gran//2)*dist, axis=0)
    y = np.dot(y_im.reshape(l), k)
    r2[i] = y
    forward_super_imposed2 += y_im

  for i in range(rot_):
    r_im = fonts.RotateImage(forward_super_imposed2, i*deg)
    r = np.dot(r_im.reshape(l), k)
    r3[i] = r

  back_super_imposed2 = np.zeros(input_image.shape)
  back_super_imposed3 = np.zeros(input_image.shape)

  intermediate_template = np.dot(template1.reshape(l), k)
  # Backward prop
  for i in range(rot_):
    r_im = fonts.RotateImage(template1, i*deg)
    back_super_imposed3 += r_im
    q = fn(intermediate_template, r3[i])
    k3[i] = k3[i]*q


  intermediate_template = np.dot(back_super_imposed3.reshape(l), k)
  for i in range(gran):
    y = np.roll(back_super_imposed3, (i-gran//2)*dist, axis=0)
    back_super_imposed2 += y
    q = fn(intermediate_template, r2[i])
    k2[i] = k2[i]*q

  intermediate_template = np.dot(back_super_imposed2.reshape(l), k)
  for i in range(gran):
    q = fn(intermediate_template, r1[i])
    k1[i] = k1[i]*q

  k3 = k3/np.sum(k3)
  k2 = k2/np.sum(k2)
  k1 = k1/np.sum(k1)

  return (np.argmax(k3)*deg, (np.argmax(k2)-gran//2)*dist, (np.argmax(k1)-gran//2)*dist)

# Number of outcomes

# test parameters:
num_angles = 10 # How many angles to test
gran = 5 # How many translations to test
distance = 2 # Distance between translations
trial_times = 1 # Repeat
d_size = 4000

r_total = 0
rot_correct = 0
x_correct = 0
y_correct = 0

print ("Total potential outcomes: %d", num_angles*gran**2)

for k in range(num_angles):
  rot = -360./num_angles * k
  rot_im = fonts.RotateImage(roman_alphabet[16], rot)
  for _j in range(gran):
    for _i in range(gran):
      for _k in range(trial_times):
        ytranslate = ((_j-gran//2)*distance)
        xtranslate = ((_i-gran//2)*distance)
        im = np.roll(rot_im, xtranslate, axis=1)
        im = np.roll(im, ytranslate, axis=0)
        # rot = MSC(seventyfive, roman_alphabet[0])
        out = X_MSC_MSC(im, roman_alphabet[16], d_size, num_angles, gran, distance)
        pred_rot = out[0]
        pred_y = out[1]
        pred_x = out[2]
        r_total += 1

        if -pred_rot == rot:
          rot_correct += 1
        if pred_y == -ytranslate:
          y_correct += 1
        if pred_x == -xtranslate:
          x_correct += 1

print "Results:"

print "Rotation_correct: "+str(float(rot_correct)/r_total)
print "Y_correct: "+str(float(y_correct)/r_total)
print "X_correct: "+str(float(x_correct)/r_total)

    
