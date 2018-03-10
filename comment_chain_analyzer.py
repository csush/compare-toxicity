import csv

oname = 'comments\\arsenal_reddit_Analyzed_overall_result.csv'
sname = 'comments\\arsenal_reddit_Analyzed_score_result.csv'

sfile = open(sname, 'rb')
ofile = open(oname, 'rb')

sreader = csv.reader(sfile, delimiter='\t')
oreader = csv.reader(ofile, delimiter='\t')

print sreader
print oreader
sreader.next()
oscore = oreader.next()
i=0
linelen = 0
for row in sreader:
    therow = row[0].split(',')
    theoverall = therow[-1]
#    print theoverall
    while oscore[0][0:min(4,len(oscore[0]))] != theoverall[0:min(4,len(theoverall))]:
        i += 1
#        print  '            ',i, '            ',oscore
        oscore = oreader.next()
        linelen = 0
    linelen += 1
    if linelen >= 3:
        print therow[0],linelen
    

sfile.close()
ofile.close()
