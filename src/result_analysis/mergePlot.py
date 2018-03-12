import csv
import os
import matplotlib.pyplot as plt
import numpy as np
def plot_hist(x, titleName):
    plt.xlim([-1, 1])
    #bins = np.arange(-1, 1, 20)
    #np.histogram(x, bins=20)
    plt.hist(x, bins = 20)
    plt.title(titleName)
    plt.xlabel("Score")
    plt.ylabel("Number of Comments")
    plt.show()

def plot_pie(sizes, titleName):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Negative -1 to -0.5', 'Negative -0.5 to 0', 'Neutral', 'Positive 0 to +0.5', 'Positive +0.5 to +1'
    explode = (0.1, 0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(titleName)
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

def myPlot(dictname, titleName):
    fileDir = dictname
    fileList = []
    score_sheet = []
    
    negative10 = 0
    negative05 = 0
    neutral = 0
    positive05 =0
    positive10 = 0
    
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
                    
                    # classify score
                    if overall_score < -0.5:
                        negative10 += 1
                    
                    elif overall_score < 0:
                        negative05 += 1
                    
                    elif overall_score == 0:
                        neutral += 1
                    
                    elif overall_score < 0.5:
                        positive05 += 1
                    
                    else:
                        positive10 += 1

    pieList = [negative10, negative05, neutral, positive05, positive10]
    plot_pie(pieList, titleName)

    #print(type(score_sheet[0]))
    #print("Data Volumn: ", len(score_sheet))
    np.array(score_sheet).astype(np.float)
    plot_hist(score_sheet, titleName)
    # return score_sheet
