import csv
import os
import matplotlib.pyplot as plt

def plot_result(x):
    import matplotlib.pyplot as plt
    plt.hist(x)
    plt.title('Histogram of score distribution')
    plt.show()

if __name__ == '__main__':
    fileDir = './cutted_overall_result/facebook'
    fileList = []
    score_sheet = []
    
    # read all filename in the directory and store in fileList as List
    for file in os.listdir(fileDir):
        name = fileDir + '/' + file
        fileList.append(name)

    # read all file in the fileList and append score to score_sheet
    for filename in fileList:
        with open (filename, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t')

            for row in spamreader:
                overall_score = row[0]
                score_sheet.append(overall_score)

    print("Data Volumn: ", len(score_sheet)) 
    plot_result(score_sheet)