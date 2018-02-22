'''
Program to generate top comments from any thread on Reddit and put it into a csv file with column name Top Comments.
'''

import praw
from praw.models import MoreComments
import csv

reddit = praw.Reddit(user_agent='CommentScraper (by /u/ConnectBad)',
                     client_id='dHUYT7-TgK2aKA', client_secret="Sg2ZreLSzOYqURbJK4QA5Ysh7tM",
                     username='ConnectBad', password='passwordfor180')

submission = reddit.submission(url='https://www.reddit.com/r/worldnews/comments/7za8v1/trump_endorses_guns_for_teachers_to_stop_shootings/')

myFile = open('all_reddit_comments.csv', 'w') 
writer = csv.writer(myFile, dialect='excel')
writer.writerow(['Top Comments'])

submission.comments.replace_more(limit=None)
for comment in submission.comments.list():
    writer.writerow([comment.body.encode('utf-8')])
