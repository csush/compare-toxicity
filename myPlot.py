import matplotlib.pyplot as plt
import csv

def plot_result(x):
    import matplotlib.pyplot as plt
    plt.hist(x)
    plt.title('Histogram of score distribution')
    plt.show()

with open('LOL_Analyzed_overall_result.csv','r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    score_sheet = []

    for row in spamreader:
        overall_score = row[0]
        score_sheet.append(overall_score)

plot_result(score_sheet)