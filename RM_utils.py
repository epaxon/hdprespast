import os
import csv
import numpy as np
import utils
import hrr_utils
import time
import random

from scipy import spatial
from scipy.spatial import distance
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

cwd = os.getcwd()

def read_verbs(url):
    keys = ['low', 'medium', 'high']
    verbs = {key: [] for key in keys}
    
    with open(url, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        curr_verbs = []
        i = 0
        for row in reader:
            if row[0] in keys:
                curr_verbs = verbs[row[0]]
            else:
                curr_verbs.append([row[0], row[1]])
            i+=1
    return verbs

def ngram_encode(ngram_str, letter_vecs, alph):
    vec = np.zeros(letter_vecs.shape[1])
    
    full_str = '#' + ngram_str + '.'
    
    
    for il, l in enumerate(full_str[:-2]):
        trivec = letter_vecs[alph.find(full_str[il]), :]
        for c3 in range(1, 3):
            trivec = trivec * np.roll(letter_vecs[alph.find(full_str[il+c3]), :], c3)
            
        vec += trivec
    return vec

def ngram_encode_cl(ngram_str, letter_vecs, alph):
    vec = ngram_encode(ngram_str, letter_vecs, alph)

    return 2* (vec + 0.1*(np.random.rand(letter_vecs.shape[1])-0.5) > 0) - 1

def genX(verbs, N, dic1, dic2, alph):
    X = np.zeros((len(verbs), N)) # Exclusively difference PAST1-PRES1
    PRES1 = np.zeros((len(verbs), N))
    PRES2 = np.zeros((len(verbs), N))
    PAST1 = np.zeros((len(verbs), N))
    PAST2 = np.zeros((len(verbs), N))
    
    for m in range(len(verbs)):
        pair = verbs[m]

        PRES1[m] = ngram_encode_cl(pair[1], dic1, alph)
        PRES2[m] = ngram_encode_cl(pair[1], dic2, alph)
        PAST1[m] = ngram_encode_cl(pair[0], dic1, alph)
        PAST2[m] = ngram_encode_cl(pair[0], dic2, alph)
#         print(pair[0], np.sum(PRES1[m]), pair[1], np.sum(PRES2[m]))
        
    #X = np.where(PAST1-PRES1 > 0, 1, -1)
    X = PAST1-PRES1
    return X, PRES1, PRES2, PAST1, PAST2

def gen_verb_set(verbs, freq_sizes, N, dic1, dic2, alph):
    keys = ['high', 'medium', 'low']
    verb_set_keys = ['dif', 'pres1', 'pres2', 'past1', 'past2']
    size = sum([freq_sizes[freq] for freq in keys])    
    verb_set = {key: np.zeros((size, N)) for key in verb_set_keys}
    i = 0
    verbs_to_encode = []
    for freq in keys:
        size = freq_sizes[freq]
        indices = list(np.random.choice(len(verbs[freq]), size, replace=False))
        verbs_to_encode += [verbs[freq][index] for index in indices]
        
    verb_set['dif'], verb_set['pres1'], verb_set['pres2'], verb_set['past1'], verb_set['past2'] = genX(verbs_to_encode, N, dic1, dic2, alph)

    return verb_set       
        
def train(tv, past, present):
    tv += np.multiply(past, present)
    return tv

def reg_train(tv, past, present, sim, N):
    pred = np.multiply(tv, present)
    #pred = np.where(pred>0, 1, -1)
    #print (sim(pred, past)),
    tv += ((N-sim(pred, past))/float(N)) * np.multiply(past, present)
    return tv

def train_diff(tv, past2, present1, present2, N):
    tv += np.multiply(present1, past2-present2)
    return tv

def reg_train_diff(tv, past2, present1, present2, N):
    pred = np.multiply(tv, present1) + present2
    #pred = np.where(pred>0, 1, -1)
    #print (sim(pred, past)),
    tv += ((N-sim(pred, past2))/float(N)) * np.multiply(past2-present2, present1)
    return tv

def outer_train(W, past, present, N):
    # col x row
    W += np.outer(present, past)
    return W

def outer_reg_train(W, Past, Present, sim, N):
    pred = np.dot(W, Present)
    W += ((N*N-sim(pred, Past))/float(N*N)) * np.outer(Past, Present)
    return W

def outer_train_diff(W, past2, present1, present2):
    W += np.outer(present1, past2-present2)
    return W

def outer_reg_train_diff(W, past2, present1, present2, sim, N):
    #W.T.dot(trainpres1[:k].T).T + trainpres2[:k]*N
    pred = np.dot(W, present1) + present2*N
    W += ((N*N-sim(pred, past2))/float(N*N)) * np.outer(present1, past2-present2)
    return W

def sim(x, y):
    if len(x.shape) == 1 or len(y.shape)==1:
        return np.dot(x, y)
    return np.sum(np.multiply(x, y), axis=1)

def compare(pred, past2):
    acc = 0
    indices = []
    for i in range(pred.shape[0]):
        sims = sim(past2, pred[i])
        predi = np.argmax(sims)
        acc += i == predi
        indices.append((i, predi))
    return acc/float(past2.shape[0]), indices
        
def round_to_tick(number):
    """Round a number to the closest half integer."""
    return round(number * 2) / 2

def graph_onetype(x, y, ystd, accuracy, title=None, xlabel='number of words', ylabel='average dot product', legend='upper left'):   
    plt.figure(1)
    plt.subplot(211)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    start = round_to_tick(min(y) - max(ystd))
    end = round_to_tick(max(y) + max(ystd))
    
    ystd = np.clip(ystd, max(-8, start-.5), min(8, end+.5))
    plt.plot(x, y, c='b', lw=2, label='Train')
    plt.fill_between(x, y-ystd, y+ystd, facecolor='b', alpha=0.1)
    plt.legend(loc=legend,fontsize=12)

    plt.xlabel('Number Training Examples',fontsize=16)
    plt.ylabel('Feature Similarity',fontsize=16)
#     plt.title(title,fontsize=16)
    plt.tight_layout()
    
    plt.subplot(212)
    plt.plot(x, accuracy)
    plt.xlabel('Number Training Examples',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)


def graph(x, y1, y2, y1std, y2std, accuracy, irreg_accuracy, title=None, xlabel='number of words', ylabel='average dot product', legend='upper left'):   
    plt.figure(1)
    plt.subplot(211)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    
    start = round_to_tick(min(min(y1), min(y2)) - max(max(y1std),max(y2std)))
    end = round_to_tick(max(max(y1), max(y2)) + max(max(y1std),max(y2std)))
    
    y1std = np.clip(y1std, max(-8, start-.5), min(8, end+.5))
    y2std = np.clip(y2std, max(-8, start-.5), min(8, end+.5))
    

    plt.plot(x, y1, c='b', lw=2, label='Regular')
    plt.plot(x, y2, c='g', lw=2, label='Irregular')

    plt.fill_between(x, y1-y1std, y1+y1std, facecolor='b', alpha=0.1)
    plt.fill_between(x, y2-y2std, y2+y2std, facecolor='g', alpha=0.1)
    
#     plt.fill_between(x, y2-y2std, y2+y2std, color='none', alpha=0.3, hatch="////", edgecolor="g")


    plt.legend(loc=legend,fontsize=12)

    plt.xlabel('Number Training Examples',fontsize=16)
    plt.ylabel('Feature Similarity',fontsize=16)
#     plt.title(title,fontsize=16)

    plt.tight_layout()

    plt.subplot(212)
    plt.plot(x, accuracy, c='b', label='Regular')
    plt.plot(x, irreg_accuracy, c='g', label='Irregular')
    plt.xlabel('Number Training Examples',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.legend(loc=legend,fontsize=12)
    
# https://matplotlib.org/api/lines_api.html
def graph_separate(x, y1, y2, y1std, y2std, \
                   y1irregular, y2irregular, \
                   accuracy, irreg_accuracy, \
                   title=None, xlabel='number of words', ylabel='average dot product', legend='upper left'):   
    plt.figure(1)
    plt.subplot(211)
    fig = plt.figure(figsize=(4,3))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    
    start = round_to_tick(min(min(y1), min(y2)) - max(max(y1std),max(y2std)))
    end = round_to_tick(max(max(y1), max(y2)) + max(max(y1std),max(y2std)))
    
    y1std = np.clip(y1std, max(-8, start-.5), min(8, end+.5))
    y2std = np.clip(y2std, max(-8, start-.5), min(8, end+.5))
    

    plt.plot(x, y1, c='b', lw=2, label='Train')
    plt.plot(x, y2, c='g', lw=2, label='Test')
    plt.plot(x, y1irregular, ':', c='b', lw=2, label='Train irregular')
    plt.plot(x, y2irregular, ':', c='g', lw=2, label='Test irregular')

    plt.fill_between(x, y1-y1std, y1+y1std, facecolor='b', alpha=0.1)
    plt.fill_between(x, y2-y2std, y2+y2std, facecolor='g', alpha=0.1)

    plt.legend(loc=legend,fontsize=12)

    plt.xlabel('Number Training Examples',fontsize=16)
    plt.ylabel('Feature Similarity',fontsize=16)
    plt.title(title,fontsize=16)

    plt.xlim([0, 1700])

    plt.tight_layout()
    
    plt.subplot(212)
    plt.plot(x, accuracy, c='b', label='Regular')
    plt.plot(x, irreg_accuracy, c='g', label='Irregular')
    plt.xlabel('Number Training Examples',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.legend(loc=legend,fontsize=12)
    
def trigram_dict_onetype(past2, pres1, train_func, N):
    psi = np.zeros(N)
    psi = train_func(psi, past2[0], pres1[0])
    x = np.arange(1, pres1.shape[0]+1, 1)
    y = np.zeros(pres1.shape[0])
    ystd = np.zeros(pres1.shape[0])
    accuracy = np.zeros(pres1.shape[0])
    for k in range(1,past2.shape[0]):
        pred = np.multiply(psi, pres1[:k])/float(N)
        sim1 = sim(pred, past2[:k])
        
        y[k] = sim1.mean()
        ystd[k] = np.nanstd(sim1, axis=0)
        accuracy[k], indices = compare(pred, past2[:k])

        psi = train_func(psi, past2[k], pres1[k])
    return x,y,ystd,accuracy

def trigram_dict(comb_past2, comb_pres1, irreg_past2, irreg_pres1, train_func, N):
    psi = np.zeros(N)
    psi = train_func(psi, comb_past2[0], comb_pres1[0])
    random_vecs = np.random.randn(comb_past2.shape[0], N)
    x = np.arange(1, comb_pres1.shape[0]+1, 1)
    y1 = np.zeros(comb_pres1.shape[0])
    y2 = np.zeros(comb_pres1.shape[0])
    y1std = np.zeros(comb_pres1.shape[0])
    y2std = np.zeros(comb_pres1.shape[0])

    accuracy = np.zeros(comb_pres1.shape[0])
    irreg_accuracy = np.zeros(comb_pres1.shape[0])
    
    for k in range(1,comb_pres1.shape[0]):
        pred = np.multiply(psi, comb_pres1[:k])/float(N)
        irreg_pred = np.multiply(psi, irreg_pres1)
        sim1 = sim(pred, comb_past2[:k])
        sim2 = sim(irreg_pred, irreg_past2)
        
        y1[k] = sim1.mean()
        y1std[k] = np.nanstd(sim1, axis=0)
        y2[k] = sim2.mean()
        y2std[k] = sim2.std(axis=0)
        accuracy[k], indices = compare(pred, comb_past2[:k])
        irreg_accuracy[k], irreg_indices = compare(irreg_pred, irreg_past2[:k])

        psi = train_func(psi, comb_past2[k], comb_pres1[k])
    return x,y1,y2,y1std,y2std, accuracy, irreg_accuracy

def diff_trigram_dict_onetype(past2, pres1, pres2, N, train_func=train_diff):
    psi = np.zeros(N)
    psi = train_func(psi, past2[0], pres1[0], pres2[0], N)
    
    x = np.arange(1, pres1.shape[0]+1, 1)
    y = np.zeros(pres1.shape[0])
    ystd = np.zeros(pres1.shape[0])
    accuracy = np.zeros(pres1.shape[0])
    for k in range(1,past2.shape[0]):
        pred = np.multiply(psi, pres1[:k])/float(N) + pres2[:k]
        sim1 = sim(pred, past2[:k])
        
        y[k] = sim1.mean()
        ystd[k] = np.nanstd(sim1, axis=0)
        accuracy[k], indices = compare(pred, past2[:k])

        psi = train_func(psi, past2[k], pres1[k], pres2[k], N)
    return x,y,ystd,accuracy

def diff_trigram_dict(comb_past2, comb_pres1, comb_pres2, irreg_past2, irreg_pres1, irreg_pres2, N, train_func=train_diff):
    psi = np.zeros(N)
    psi = train_func(psi, comb_past2[0], comb_pres1[0], comb_pres2[0], N)

    x = np.arange(1, comb_pres1.shape[0]+1, 1)
    y1 = np.zeros(comb_pres1.shape[0])
    y2 = np.zeros(comb_pres1.shape[0])
    y1std = np.zeros(comb_pres1.shape[0])
    y2std = np.zeros(comb_pres1.shape[0])

    accuracy = np.zeros(comb_pres1.shape[0])
    irreg_accuracy = np.zeros(comb_pres1.shape[0])

    for k in range(1,comb_past2.shape[0]):
        pred = np.multiply(psi, comb_pres1[:k]) + comb_pres2[:k]
        irreg_pred = np.multiply(psi, irreg_pres1) + irreg_pres2

        sim1 = sim(pred, comb_past2[:k])
        sim2 = sim(irreg_pred, irreg_past2)

        y1[k] = sim1.mean()
        y2[k] = sim2.mean()
        y1std[k] = np.nanstd(sim1, axis=0)
        y2std[k] = np.nanstd(sim2, axis=0)
        accuracy[k], indices = compare(pred, comb_past2[:k])
        irreg_accuracy[k], irreg_indices = compare(irreg_pred, irreg_past2[:k])

        psi = train_func(psi, comb_past2[k], comb_pres1[k], comb_pres2[k], N)
    return x,y1,y2,y1std,y2std, accuracy, irreg_accuracy