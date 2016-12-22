import nltk
import csv
a = nltk.corpus.brown.tagged_words()
def get_all_words(tense, dest_name):
    wordset = {}
    for elem in a:
        #print str(elem[1])
        if str(elem[1]) == tense:
            word = str(elem[0]).lower()
            if word not in wordset:
                wordset[word] = 1
            else:
                wordset[word] += 1

    keys = sorted(wordset.keys())
    with open(dest_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for k in keys:
            writer.writerow([k, wordset[k]])
    return wordset

def merge(hist1, hist2, dest_name):
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
    keys = sorted(wordset.keys())
    with open(dest_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for k in keys:
            writer.writerow([k, wordset[k]])
    return wordset


vb = get_all_words("VB", "VB.csv")
vbd = get_all_words("VBD", "VBD.csv")
vbz = get_all_words("VBZ", "VBZ.csv")

vbzwithouts = {}
for k, v in vbz.items():
    vbzwithouts[k[:len(k)-1]] = v

vbvbd = merge(vb, vbzwithouts, "VBVBZ.csv")