import csv
import os
import matplotlib.pyplot as plt
import numpy as np
def plot_result(x):
    plt.xlim([-1, 1])
    #bins = np.arange(-1, 1, 20)
    #np.histogram(x, bins=20)
    plt.hist(x, bins = 20)
    plt.title("News Comments Score")
    plt.xlabel("Score")
    plt.ylabel("Number of Comments")
    plt.show()
    
def mycsv_reader(csv_reader): 
  while True: 
    try: 
      yield next(csv_reader) 
    except csv.Error: 
      # error handling what you want.
      pass
    continue 
  return

def myPlot(dictname):
    fileDir = dictname
    fileList = []
    score_sheet = []
    
    # read all filename in the directory and store in fileList as List
    for file in os.listdir(fileDir):
        name = fileDir + '/' + file
        fileList.append(name)

    # read all file in the fileList and append score to score_sheet
    for filename in fileList:
        i = 0;
        with open (filename, 'rb') as csvfile:
            spamreader = mycsv_reader(csv.reader(csvfile, delimiter='\t'))
            print(filename)
            for row in spamreader:
                if i == 0:
                    i = 1;
                else:
                    overall_score = float(row[0])
                    #print(overall_score)
                    score_sheet.append(overall_score)
               
    print(type(score_sheet[0]))
    #print("Data Volumn: ", len(score_sheet))
    np.array(score_sheet).astype(np.float)
    plot_result(score_sheet)
    #return score_sheet
