import numpy as np
import string

def GenerateDictionary(alpha, n=10000):
  l = {}
  for i in alpha:
    l[i] = np.where(np.random.rand(n) < .5, 1, -1)
  return l

def GenerateDefaultDictionary(n=10000):
  alphabet = string.ascii_lowercase+"."+"#"
  return GenerateDictionary(alphabet, n)

def GetRV(string, dic, ngram=3):
  string = "#"+string+"."
  assert len(string) >= ngram
  
  n = dic.values()[0].shape[0]
  tv = np.zeros(n)
  for i in range(len(string)-ngram+1):
    s = string[i:i+ngram]
    l = np.ones(n)
    for j in s[::-1]:
      l = np.roll(l, 1)
      l = np.multiply(l, dic[j])
        
    tv += l
  return tv

def GetRVClip(string, dic, ngram=3, fn=GetRV):
  tv = fn(string,dic,ngram)
  return np.where(tv>0, 1, -1)

def GetRVClipHash(string, dic, ngram=3, fn=GetRV):
  tv = fn(string,dic,ngram)
  sh = bin(hash(string))[2:]
  for i in range(tv.shape[0]):
    if tv[i] == 0:
      TF = sh[i%len(sh)] == '1'
      tv[i] = TF * 1 + -1 * (1- TF)
  return np.where(tv>0, 1, -1) 

def GetAllRVGrams(string, dic, uptograms=3):
  string = "#"+string+"."
  assert len(string) >= uptograms

  n = dic.values()[0].shape[0]
  tv = np.zeros(n)
  for ngram in range(1, uptograms+1):
    for i in range(len(string)-ngram+1):
      s = string[i:i+ngram]
      l = np.ones(n)
      for j in s:
        l = np.roll(l, 1)
        l = np.multiply(l, dic[j])
          
      tv += l
  return tv
