import numpy as np

def GetRegularVerbs(frequency=False):
  """
  Returns regular verbs from cleaned/regular_verbs_clean.csv

  present, past
  """
  present = []
  past = []
  freq = []


  f = open('data/cleaned/regular_verbs_clean.csv', 'r')
  w = f.read().strip().split("\n")
  for i in w:
    k = i.split(",")
    present.append(k[0])
    past.append(k[1])
    freq.append(int(k[2]))

  if frequency:
    return present, past, np.array(freq)

  return present, past

def GetIrregularVerbs(frequency=False):
  present = []
  past = []
  freq = []
  f = open('data/cleaned/irregular_verbs_clean.csv', 'r')
  w = f.read().strip().split("\n")
  for i in w:
    k = i.split(",")
    present.append(k[0])
    past.append(k[1])
    freq.append(int(k[2]))

  if frequency:
    return present, past, np.array(freq)

  return present, past

def Shuffle(present, past, freq=None):
  if type(freq) != type(None):
    l = zip(present, past, freq)
    np.random.shuffle(l)
    present, past, freq = zip(*l)
    return present, past, freq
  l = zip(present, past)
  np.random.shuffle(l)
  present, past = zip(*l)
  return present, past


def LoadLatentSpace():
  f = open('cleaned/latent_space.txt')
  return f.read().strip().split("\n")

def LoadLatentSpaceGroups():
  f = open('cleaned/latent_grouped.txt')
  k = []
  grouping = []
  grouped = f.read().strip().split("\n")
  for line in grouped:
    if len(line) == 0 or line[0] == "#":
      continue
    if line[0] == "!":
      if len(grouping) > 0:
        k.append(grouping)
      grouping = []
      continue
    tokens = line.split(",")
    grouping.append((tokens[0], tokens[1]))

  if len(grouping) > 0:
    k.append(grouping)

  return k



