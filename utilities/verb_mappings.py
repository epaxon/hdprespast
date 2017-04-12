import random_idx
import utils
import numpy as np
import string
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys

def word_vec(word, alphabet, RI_letters, cluster_size=3, N=1000, ordered=1):
    """
    word vectors encoded as a sum of their trigrams. Thus a word vector is just like a language
    vector of yore, with a word of L letters + two spaces yielding L trigrams.
    
      j u
    j u m
    u m p
    m p 

    """
    word = " " + word + " "
    wordv = np.zeros((1,N))
    for i in range(len(word)-cluster_size):
        ngram = word[i:i+cluster_size]
        wordv += random_idx.id_vector(N, ngram, alphabet, RI_letters,ordered)
    return wordv

def mapping_vec(pres, past):
    """
    form a mapping vector from a present-tense verb to a past-tense verb by multiplying 
    the two word vectors. This same mapping vector will turn the past-tense verb to the 
    present-tense verb.  These mappings are not exact because the word vectors are
    not binary +1s and -1s.
    """
    return pres*past

def tense_vec(words, alphabet, RI_letters, cluster_size=3, N=1000, ordered=1):
    """
    adds all word vecs of the same tense together. 
    """
    tense = np.zeros((1,N))
    for word in words:
        tense += word_vec(word, alphabet, RI_letters, cluster_size, N, ordered)
    return tense

def master_map(present_tense, past_tense):
    """
    multiplies present_tense vector with past_tense vector
    """
    return present_tense*past_tense