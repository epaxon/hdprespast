import numpy as np
import math
import matplotlib.pyplot as plt
import string

N = 1000
alphabet = string.ascii_lowercase + '#' + '.'
D = len(alphabet)
z = np.ones(N)
#words = ["soberize", "revegetate", "finess", "impersonalize", "cuss", "overage", "retailor"]
#past_tense = ["soberized", "revegetated", "finessed", "impersonalized", "cussed", "overaged", "retailored"]
# past is encoded
past_tense = open("wickle_train/ed1000.txt", 'r').read().strip().split()#[0:200]
words = [pt[:len(pt)-2] for pt in past_tense]

test_past_tense = past_tense[:100]
test_words = words[:100]

past_tense = past_tense[100:]
words = words[100:]

RI = np.random.rand(D, N)
RI = np.where(RI>0.5, 1, -1)
RI = RI/float(math.sqrt(N))
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

    return 2*(z + 0.1*(np.random.rand(N) - 0.5) > 0) - 1#np.where(z/count > 0, 1, -1)

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


total_vec = np.zeros(N) #psi
seen = []
for i in range(len(words)):
    enc = encode(words[i], 3)
    enc_past = encode(past_tense[i], 3)
    total_vec += np.multiply(enc, enc_past)
    seen.append((enc, enc_past))

plt.figure()
plt.plot(enc[:50])

"""
Psi = A x B
A dot Psi x B = 1
AA=1
"""
# need to take the average similarity
num_iters = 1
data1, data3 = np.zeros((num_iters, len(seen))), np.zeros((num_iters, len(seen)))
total1, total3, count = 0, 0, 0
for i in range(num_iters):
    for j in range(len(seen)):
        pair = seen[j]
        #sim = np.dot(normalize(np.multiply(total_vec, pair[0])), pair[1])
        sim = np.dot(np.multiply(total_vec, pair[0]), pair[1])
        #sim = sim*(sim>0)
        total1 += sim

        #sim = np.dot(normalize(np.multiply(total_vec, encode("nothere", 3))), pair[0])
        sim = np.dot(np.multiply(total_vec, encode("nothere", 3)), pair[0])
        # sim = sim*(sim>0)
        total3+= sim

        count += 1
        data1[i,j] = total1/float(count)
        data3[i,j] = total3/float(count)

data1 = data1.mean(axis=0)
data3 = data3.mean(axis=0)
print data1.shape
print data3.shape

plt.figure()
plt.plot(data1, label="Average pairwise similarity")
plt.ylabel("Average similarity per vector")
plt.xlabel("Number of words in model vector")
plt.plot(data3, label="Random pairwise similarity")
plt.legend(loc=1,prop={'size':10})
plt.show()

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
"""
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
"""