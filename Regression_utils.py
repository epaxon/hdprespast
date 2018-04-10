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

def closed_train(Past, Present, rcond=1e-15): # 5e-2
    return np.dot(np.linalg.pinv(Present,rcond), Past)

def round_to_tick(number):
    """Round a number to the closest half integer."""
    return round(number * 2) / 2

# https://matplotlib.org/api/lines_api.html
def graph(x, y1, y2, y1std, y2std, \
                   yreg, yirreg, yregstd, yirregstd, \
                   acc1, acc2, accreg, accirreg, \
                   title=None, xlabel='number of words', ylabel='average dot product', legend='upper left'):
    
    plt.figure(1,figsize=(8,6))
    plt.subplot(211)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    
    start = round_to_tick(min(min(y1), min(y2)) - max(max(y1std),max(y2std)))
    end = round_to_tick(max(max(y1), max(y2)) + max(max(y1std),max(y2std)))
    
    y1std = np.clip(y1std, max(-8, start-.5), min(8, end+.5))
    y2std = np.clip(y2std, max(-8, start-.5), min(8, end+.5))
    

    plt.plot(x, y1, c='c', lw=2, label='Train')
    plt.plot(x, y2, c='m', lw=2, label='Test')
    plt.plot(x, yreg, c='y', lw=2, label='Test regular')
    plt.plot(x, yirreg, c='k', lw=2, label='Test irregular')

    plt.fill_between(x, y1-y1std, y1+y1std, facecolor='c', alpha=0.1)
    plt.fill_between(x, y2-y2std, y2+y2std, facecolor='m', alpha=0.1)
    plt.fill_between(x, yreg-yregstd, yreg+yregstd, facecolor='y', alpha=0.1)
    plt.fill_between(x, yirreg-yirregstd, yirreg+yirregstd, facecolor='k', alpha=0.1)

    plt.legend(loc=legend,fontsize=12)

    plt.xlabel('Number Training Examples',fontsize=16)
    plt.ylabel('Feature Similarity',fontsize=16)
    plt.title(title,fontsize=16)

    plt.xlim([0, 1700])

    plt.tight_layout()

    plt.subplot(212)
    plt.plot(x, acc1, c='c', label='Train')
    plt.plot(x, acc2, c='m', label='Test')
    plt.plot(x, accreg, c='y', label='Regular')
    plt.plot(x, accirreg, c='k', label='Irregular')
    plt.xlabel('Number Training Examples',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.legend(loc=legend,fontsize=12)

# one type could be by changing trainpres1, trainpast2 to combo, reg, or irreg
def closed_trigram_dict(trainpres1, testpres1, trainpast2, testpast2, \
                                regtestpres1, irregtestpres1, regtestpast2, irregtestpast2, \
                                N, rcond=1e-15): # 5e-2
    # closed_train(Past, Present, rcond=1e-15): # 5e-2
    x = np.arange(1, trainpres1.shape[0], 10)
    y1 = np.zeros(x.shape[0])
    y2 = np.zeros(x.shape[0])
    yreg = np.zeros(x.shape[0])
    yirreg = np.zeros(x.shape[0])
    
    y1std = np.zeros(x.shape[0])
    y2std = np.zeros(x.shape[0])
    yregstd = np.zeros(x.shape[0])
    yirregstd = np.zeros(x.shape[0])
    
    acc1 = np.zeros(x.shape[0])
    acc2 = np.zeros(x.shape[0])
    accreg = np.zeros(x.shape[0])
    accirreg = np.zeros(x.shape[0])

    for i in range(x.shape[0]):#trainpres1.shape[0]):
        k = x[i]
        W = closed_train(trainpast2[:k], trainpres1[:k], rcond)
        train_pred = trainpres1[:k].dot(W)
        test_pred = testpres1.dot(W)
        regtest_pred = regtestpres1.dot(W)
        irregtest_pred = irregtestpres1.dot(W)

        sim1 = sim(train_pred, trainpast2[:k])
        sim2 = sim(test_pred, testpast2)
        sim3 = sim(regtest_pred, regtestpast2)
        sim4 = sim(irregtest_pred, irregtestpast2)

        y1[i] = sim1.mean()/N
        y2[i] = sim2.mean()/N
        yreg[i] = sim3.mean()/N
        yirreg[i] = sim4.mean()/N
        
        y1std[i] = np.nanstd(sim1, axis=0)/N
        y2std[i] = np.nanstd(sim2, axis=0)/N
        yregstd[i] = np.nanstd(sim3, axis=0)/N
        yirregstd[i] = np.nanstd(sim4, axis=0)/N
        
        acc1[i], indices = compare(train_pred, trainpast2[:k])
        acc2[i], indices = compare(test_pred, testpast2)
        accreg[i], indices = compare(regtest_pred, regtestpast2)
        accirreg[i], indices = compare(irregtest_pred, irregtestpast2)
        
    return x, y1, y2, yreg, yirreg, y1std, y2std, yregstd, yirregstd, acc1, acc2, accreg, accirreg
    

