'''
Program to generate top comments from any thread on Reddit and put it into a csv file with column name Top Comments.
'''

import praw
from praw.models import MoreComments
import csv

reddit = praw.Reddit(user_agent='CommentScraper (by /u/ConnectBad)',
                     client_id='dHUYT7-TgK2aKA', client_secret="Sg2ZreLSzOYqURbJK4QA5Ysh7tM",
                     username='ConnectBad', password='passwordfor180')

submission = reddit.submission(url='https://www.reddit.com/r/funny/comments/3g1jfi/buttons/')

myFile = open('topcomments.csv', 'w') 
writer = csv.writer(myFile, dialect='excel')
writer.writerow(['Top Comments'])

for top_level_comment in submission.comments:
    if isinstance(top_level_comment, MoreComments):
        continue
    writer.writerow([top_level_comment.body])