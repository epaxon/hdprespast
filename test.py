import numpy as np

import matplotlib.pyplot as plt
import string

N = 5
alphabet = string.ascii_lowercase + '#' + '.'
D = len(alphabet)
z = np.ones(N)

words = open("words", 'r').read().strip().split()[0:200]
past_tense = [word[:len(words)-2] for word in words]


RI = np.random.rand(D, N)
RI = np.where(RI>0.5, 1, -1)

def encode(s, cluster_size):
  s = "#"+s+"."
  global RI
  global N
  global D
  global alphabet
  z = np.zeros(N)
  count = 0
  for i in range(len(s)-cluster_size):
    count += 1
    l = np.ones(N)
    for j in s[i:i+cluster_size]:
      l = np.multiply(RI[alphabet.index(j), :], l)
      l = np.roll(l, 1)
    z += l

  return z/count

X = np.zeros((N, len(words)))
Y = np.zeros(X.shape)

for k in range(len(words)):
  enc = encode(words[k], 3)
  enc_past = encode(past_tense[k], 3)
  X[:, k] = enc
  Y[:, k] = enc_past

W =  np.dot(np.linalg.pinv(X.T), Y.T)

# total = 0
# for k in range(len(words)):
#   enc = encode(words[k], 3)
#   enc_past = encode(past_tense[k], 3)
#   l = np.dot(W, enc)
#   sim = np.dot(l, enc_past)
#   total+= sim

# print total/len(words)

# total = 0
# for k in range(len(words)):
#   enc = encode(words[k], 3)
#   enc_past = encode("nothere", 3)
#   l = np.dot(W, enc)
#   sim = np.dot(l, enc_past)
#   total+= sim

# print total/len(words)


# Binding

# normalize = lambda x: (x-np.mean(x))/np.std(x)

# data1 = [0]
# data3 = [0]
# total_vec = np.zeros(N)
# seen = []
# for k in range(len(words)):
#   enc = encode(words[k], 3)
#   enc_past = encode(past_tense[k], 3)
#   total_vec += np.multiply(enc, enc_past)
#   seen.append((enc, enc_past))

#   if k%100 == 1:
#     print k

#   total1 = 0
#   total3 = 0
#   count = 0
#   for pair in seen:
#     sim = np.dot(normalize(np.multiply(total_vec, pair[0])), pair[1])
#     # sim = sim*(sim>0)
#     total1 += sim

#     sim = np.dot(normalize(np.multiply(total_vec, encode("nothere", 3))), pair[0])
#     # sim = sim*(sim>0)
#     total3+= sim

#     count += 1
#   data1.append(total1/count)
#   data3.append(total3/count)


# plt.plot(data1, label="Average pairwise similarity")
# plt.ylabel("Average similarity per vector")
# plt.plot(data3, label="Random pairwise similarity")
# plt.show()

# total = 0
# for k in range(len(words)):
#   enc = encode(words[k], 3)
#   enc_past = encode(past_tense[k], 3)
  
#   sim = np.dot(np.multiply(total_vec, enc), enc_past)
#   total+= sim

# print total/len(words)

# total = 0
# for k in range(len(words)):
#   enc = encode(words[k], 3)
#   enc_past = encode("nothere", 3)

#   sim = np.dot(np.multiply(total_vec, enc), enc_past)
#   total+= sim

# print total/len(words)
data = [0, 0, 0, 0, 0]
for i in range(5, 50):
  RI = np.random.rand(D, N)
  RI = np.where(RI>0.5, 1, -1)
  total_vec = np.zeros(N)
  for j in range(len(words)):
    enc = encode(words[j], 3)
    enc_past = encode(past_tense[j], 3)
    total_vec += np.multiply(enc, enc_past)

  sim = 0
  for j in range(len(words)):
    enc = encode(words[j], 3)
    enc_past = encode(past_tense[j], 3)








