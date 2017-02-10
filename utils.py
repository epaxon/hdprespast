import numpy as np

def GetRegularVerbs():
  """
  Returns regular verbs from cleaned/regular_verbs_clean.csv

  present, past
  """
  present = []
  past = []

  f = open('cleaned/regular_verbs_clean.csv', 'r')
  w = f.read().strip().split("\n")
  for i in w:
    k = i.split(",")
    present.append(k[0])
    past.append(k[1])

  return present, past

def GetIrregularVerbs():
  present = []
  past = []
  f = open('cleaned/irregular_verbs_clean.csv', 'r')
  w = f.read().strip().split("\n")
  for i in w:
    k = i.split(",")
    present.append(k[0])
    past.append(k[1])

  return present, past

def Shuffle(present, past):
  l = zip(present, past)
  np.random.shuffle(l)
  present, past = zip(*l)
  return list(present), list(past)

def LoadLatentSpace():
  f = open('cleaned/latent_space.txt')
  return f.read().strip().split("\n")
