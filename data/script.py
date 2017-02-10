import urllib2

f = open("VB.csv", "r").read().strip().split('\n')
g = open("tense.csv", "w")

for line in f:
    k = line.strip().split(",")
    word = k[0]

    resp = urllib2.urlopen("http://www.wordhippo.com/what-is/the-past-tense-of/%s.html"%word).read()
    split = resp.split("The past tense of ")

    if len(split) == 2:
        print "nothing for %s"%word
        continue

    try:
        past = split[2].split("b>")[1][:-2]
        print past
        g.write(k[0]+","+past+","+k[2]+"\n")
    except:
        print "Failed %s"%word

