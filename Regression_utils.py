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


def spell(coef_hists):
    pred = ''
    alphis = []

    for i in range(len(coef_hists)):
        x, alphi = np.unravel_index(coef_hists[i].argmax(), coef_hists[i].shape)
        pred += alph[alphi]
        alphis.append(alphi)

    return pred, alphis

def explain_away(bound_vec, letter_vecs, state_length=5,  max_steps=500, conseq=100):
    th_vec = bound_vec.copy()
    conseq_preds = []
    convstep = -1
    
    states, coef_hists = initialize(letter_vecs, state_length, max_steps)
    
    #state_length = len(states)
    N = letter_vecs.shape[1]
    D = letter_vecs.shape[0]

    for i in range(max_steps):

        for j in range(1, state_length-1):
            coef_hists[j-1][i, :] = np.dot(letter_vecs, states[j])
        
        for j in range(1, state_length-1):
            mxidx = np.argmax(np.abs(coef_hists[j-1][i,:]))
            states[j] *= np.sign(coef_hists[j-1][i, mxidx])
        ljds = []
        for j in range(1, state_length-1):
            if j == 1:
                ljds.append(
                    (np.roll(th_vec * states[0] * np.roll(states[2], 2), -1) +
                    th_vec * np.roll(states[2], 1) * np.roll(states[3], 2)) / 2
                )
            elif 1 < j < state_length-2:
                ljds.append(
                    (np.roll(th_vec * states[j-2] * np.roll(states[j-1], 1), -2) +
                    np.roll(th_vec * states[j-1] * np.roll(states[j+1], 2), -1) +
                      th_vec * np.roll(states[j+1], 1) * np.roll(states[j+2], 2)) / 3
                )
            else:
                ljds.append(
                    (np.roll(th_vec * states[j-1] * np.roll(states[j+1], 2), -1) +
                   np.roll(th_vec * states[j-2] * np.roll(states[j-1], 1), -2)) / 2
                )

        for j in range(1, state_length-1):    
            states[j] = 1.2*np.dot(letter_vecs.T, np.dot(ljds[j-1], letter_vecs.T)/N) + states[j]

            gg=10000
            states[j] = gg*np.tanh(states[j]/gg)
            #states[j] = 2.0 * (states[j] > 0) - 1

        bv = states[0] * np.roll(states[1],1) * np.roll(states[2],2)  
        for j in range(1, state_length-2):
            bv += states[j] * np.roll(states[j+1],1) * np.roll(states[j+2],2) 
            
        th_vec = bound_vec - bv
        pred, alphis = spell(coef_hists)
        
        if convstep == -1:
            if len(conseq_preds) == conseq and len( set( conseq_preds ) ) == 1:
                convstep = i
                #break
            conseq_preds.append(pred)
            conseq_preds = conseq_preds[-conseq:]

#         print ('pred', pred)
#     print ('conseq', conseq_preds, len(conseq_preds))
#     print ('breaked', i, convstep)
    if convstep == -1:
        convstep=i
    return states, coef_hists, convstep
    

def resplot(word_length, states, coef_hists, N, nsteps, start):
    
    pred, alphis = spell(coef_hists)
    print pred
    rows = 1
    columns = word_length

    fig, axes = plt.subplots(rows, columns, sharex='all', squeeze=True, figsize=(6, 2.5))
    cols = get_cmap('copper', min(500,n_steps))
    x = np.linspace(0,len(alph)-2,len(alph)-2)
    labels = list(alph)
    plt.xticks(x, labels)
    
    
    
    for j in range(word_length):
        for i in range(start, min(500,n_steps)):
            # graphing the max positive at every iteration is not intuitive, since we should
            # be focusing on how our predicted letter's probability increases over time
            coef_hists[j][i,alphis[j]] = np.abs(coef_hists[j][i,alphis[j]])
            axes[j].plot(coef_hists[j][i,:], lw=1.7, c=cols(i))
            
        step, alphi = np.unravel_index(coef_hists[j].argmax(), coef_hists[j].shape)
        axes[j].plot(alphi, coef_hists[j][step, alphi], '+')

    #plt.savefig('figures/'+title+pred+'-N='+str(N)+'-steps='+str(nsteps)+'-reg='+reg+'.svg')
    
def accuracy(pred, actual):
    acc = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            acc += 1
    return acc / float(len(pred))

def fill_run(words, word_length=8, trials=1, N=1000, n_steps=500):
    all_coefs = []
    all_convcoefs = []
    all_words = []
    all_convwords = []
    for trial in range(trials):
        for word in words:
            dic1 = 2 * (np.random.randn(D, N) < 0) - 1

            word_vec = ngram_encode_cl(word, dic1, alph)

            states, coef_hists = initialize(dic1, word_length+2, n_steps)
            states, coef_hists, steps = explain_away(
                word_vec, states, coef_hists, word_length+2, dic1, N, D, n_steps)
            pred, alphis = spell(coef_hists)

            conv_coef_hists = [coef_hists[j][:steps] for j in range(len(coef_hists))]
            predconv, alphisconv = spell(conv_coef_hists)
            #print (pred, predconv, word)
            if pred == word:
                all_coefs.append(coef_hists)
                all_words.append(word)
            if predconv == word:
                all_convcoefs.append(conv_coef_hists)
                all_convwords.append(word)
    return all_coefs, all_words, all_convcoefs, all_convwords
            

def initialize(letter_vecs, state_length=5, n_steps=500):
    
    N = letter_vecs.shape[1]
    D = letter_vecs.shape[0]
    
    states = []
    coef_hists = []
    
    for i in range(state_length):
        states.append(np.random.randn(N))
    
    for i in range(1, state_length-1):
        states[i] = np.dot(letter_vecs.T, np.dot(states[i], letter_vecs.T))

    for i in range(1, state_length-1):
        states[i] = states[i]/norm(states[i])

    states[0] = letter_vecs[alph.find('#'), :]
    states[state_length-1] = letter_vecs[alph.find('.'), :]
    
    for i in range(1, state_length-1):
        coef_hists.append(np.zeros((n_steps, D)))
    
    return states, coef_hists


def resonate_trigram(bound_vec, letter_vecs, state_length=5,  max_steps=500):
    th_vec = bound_vec.copy()
    conseq_preds = []
    convstep = max_steps-1
    
    states, coef_hists = initialize(letter_vecs, state_length, max_steps)
    
    #state_length = len(states)
    N = letter_vecs.shape[1]
    D = letter_vecs.shape[0]

    for i in range(max_steps):
        all_converged = np.zeros(state_length-2)
        for j in range(1, state_length-1):
            coef_hists[j-1][i, :] = np.dot(letter_vecs, states[j])
        
            if i > 1:
                all_converged[j-1] = np.allclose(coef_hists[j-1][i,:], coef_hists[j-1][i-1, :],
                                                atol=5e-3, rtol=2e-2)
                
            mxidx = np.argmax(np.abs(coef_hists[j-1][i,:]))
            #states[j] *= np.sign(coef_hists[j-1][i, mxidx])
            
        if np.all(all_converged):
            convstep=i
            print 'converged:', i,
            break
        
        ljds = []
        for j in range(1, state_length-1):
            if j == 1:
                ljds.append(
                    (np.roll(th_vec * states[0] * np.roll(states[2], 2), -1) +
                    th_vec * np.roll(states[2], 1) * np.roll(states[3], 2)) / 2
                )
            elif 1 < j < state_length-2:
                ljds.append(
                    (np.roll(th_vec * states[j-2] * np.roll(states[j-1], 1), -2) +
                    np.roll(th_vec * states[j-1] * np.roll(states[j+1], 2), -1) +
                      th_vec * np.roll(states[j+1], 1) * np.roll(states[j+2], 2)) / 3
                )
            else:
                ljds.append(
                    (np.roll(th_vec * states[j-1] * np.roll(states[j+1], 2), -1) +
                   np.roll(th_vec * states[j-2] * np.roll(states[j-1], 1), -2)) / 2
                )

        for j in range(1, state_length-1):    
            states[j] = np.dot(letter_vecs.T, np.dot(ljds[j-1], letter_vecs.T)/N) #+ 1.0*states[j]
            states[j] = 2.0 * (states[j] > 0) - 1
            #states[j] /= norm(states[j])
        
    return states, coef_hists, convstep
        
