import nltk
import csv
a = nltk.corpus.brown.tagged_words()

def read_csv(filepath):
    wordset = {}
    with open(filepath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            wordset[row[0]] = row[1]
    return wordset

def write_csv(dest_name, wordset):
    keys = sorted(wordset.keys())
    with open(dest_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for k in keys:
            writer.writerow([k, wordset[k]])

def get_all_words_reg(tense, dest_name):
    wordset = {}
    for elem in a:
        #print str(elem[1])
        if str(elem[1]) == tense:
            word = str(elem[0]).lower()
            if word not in wordset:
                wordset[word] = 1
            else:
                wordset[word] += 1
    write_csv(dest_name, wordset)
    return wordset

def merge_reg(hist1, hist2, dest_name):
    wordset = {}
    for k, v in hist1.items():
        if k not in wordset.keys():
            wordset[k] = v
        else:
            wordset[k] += v
    for k, v in hist2.items():
        if k not in wordset.keys():
            wordset[k] = v
        else:
            wordset[k] += v
    write_csv(dest_name, wordset)
    return wordset

"""
find matched by going back at least 2 characters. 
max 4 characters. datasets broken rip
"""
def match_reg(present, past):
    pastprespair = {}
    pastrem2, pastrem3 = {}, {}
    presrem1 = {} # idk if this one is necessary
    for k in past.keys():
        pastrem2[k[:len(k)-2]] = k
        pastrem3[k[:len(k)-3]] = k
    for k in present.keys():
        presrem1[k[:len(k)-1]] = k

    processedset = set()
    for k, v in present.items():

        if k in pastrem2.keys():
            pastprespair[pastrem2[k]] = k
        elif k[:len(k)-1] in pastrem2.keys():
            pastprespair[pastrem2[k[:len(k)-1]]] = k
        elif k in pastrem3.keys():
            pastprespair[pastrem3[k]] = k
        elif k[:len(k)-1] in pastrem3.keys():
            pastprespair[pastrem3[k[:len(k)-1]]] = k
   
    pres, pas = {}, {}
    for k, v in pastprespair.items():
        pas[k] = past[k]
        pres[v] = present[v]
    write_csv("past_reg.csv", pas)
    write_csv("present_reg.csv", pres)
    return pas, pres
    
"""
vb = get_all_words_reg("VB", "VB.csv")
vbd = get_all_words_reg("VBD", "VBD.csv")
vbz = get_all_words_reg("VBZ", "VBZ.csv")

vbzwithouts = {}
for k, v in vbz.items():
    vbzwithouts[k[:len(k)-1]] = v

vbvbd = merge_reg(vb, vbzwithouts, "VBVBZ.csv")
"""

past = read_csv("VBD.csv")
present = read_csv("VBVBZ.csv")
pas, pres = match_reg(present, past)
