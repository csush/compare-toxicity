import csv
import os
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# initalize analyzer as SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# open csv file for output
with open('Analyzed_result2.csv', 'w') as csvfile:
	spamwriter = csv.DictWriter(csvfile,fieldnames=['neg', 'neu', 'compound', 'pos'])
	spamwriter.writeheader()

	# open csv file for input
	with open('DonaldTrump_facebook_comments.csv', 'r') as csvfile:
		comments = csv.DictReader(csvfile)

		for comment in comments:
			# For DEBUG - print current comment been process
			print(" ")
			print(comment['message'])

			# analyzing with the analyzer and record as result
			result = analyzer.polarity_scores(comment['message'])

			for r in result:
				# For DEBUG - print current result
				print('{0}: {1}, '.format(r, result[r]))
			
			# write result on the csv file
			spamwriter.writerow(result)

