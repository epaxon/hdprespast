import csv
import numpy as np

def sort_file(filename, outfilename, std_thresh):
    pres = []
    past = []
    freqs = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pres.append(row[0])
            past.append(row[1])
            freqs.append(row[2])

    freqs = np.array(freqs).astype(np.int)
    std = np.std(freqs)
    mean = np.mean(freqs)
    sorted_words = {'low': [], 'medium': [], 'high': []}
    for i in range(freqs.shape[0]):
        if freqs[i] < mean - std*std_thresh:
            sorted_words['low'].append([pres[i], past[i], freqs[i]])
        elif mean - std*std_thresh <= freqs[i] and freqs[i] < mean + std*std_thresh:
            sorted_words['medium'].append([pres[i], past[i], freqs[i]])
        else:
            sorted_words['high'].append([pres[i], past[i], freqs[i]])

    print len(sorted_words['low'])
    print len(sorted_words['medium'])
    print len(sorted_words['high'])
    with open(outfilename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['low'])
        for row in sorted_words['low']:
            writer.writerow(row)
        writer.writerow(['medium'])
        for row in sorted_words['medium']:
            writer.writerow(row)
        writer.writerow(['high'])
        for row in sorted_words['high']:
            writer.writerow(row)

std_thresh = 1
files = ['cleaned/irregular_verbs_clean.csv', 'cleaned/regular_verbs_clean.csv']
outfiles = ['cleaned/Brown_irregular_std='+str(std_thresh)+'.csv', 'cleaned/Brown_regular_std='+str(std_thresh)+'.csv']

for i in range(len(files)):
    sort_file(files[i], outfiles[i], std_thresh)
