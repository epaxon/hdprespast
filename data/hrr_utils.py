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
    for j in s:
      l = np.roll(l, 1)
      l = np.multiply(l, dic[j])
        
    tv += l
  return tv

def GetRVClip(string, dic, ngram=3):
	tv = GetRV(string,dic,ngram)
	return np.where(tv>0, 1, -1)