import requests
import requests.auth
import praw

'''client_auth = requests.auth.HTTPBasicAuth('p-jcoLKBynTLew', 'gko_LXELoV07ZBNUXrvWZfzE3aI')
post_data = {"grant_type": "password", "username": "reddit_bot", "password": "snoo"}
headers = {"User-Agent": "ChangeMeClient/0.1 by YourUsername"}
response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
response.json()'''

username = 'ConnectBad'
userAgent = "CommentScraper/0.1 by " + username
clientId = 'dHUYT7-TgK2aKA'
clientSecret = "Sg2ZreLSzOYqURbJK4QA5Ysh7tM"
password = "passwordfor180"
r = praw.Reddit(user_agent=userAgent, client_id=clientId, client_secret=clientSecret)

def getSubComments(comment, allComments, verbose=True):
  allComments.append(comment)
  if not hasattr(comment, "replies"):
    replies = comment.comments()
    if verbose: print("fetching (" + str(len(allComments)) + " comments fetched total)")
  else:
    replies = comment.replies
  for child in replies:
    getSubComments(child, allComments, verbose=verbose)


def getAll(r, submissionId, verbose=True):
  submission = r.submission(submissionId)
  comments = submission.comments
  commentsList = []
  for comment in comments:
    getSubComments(comment, commentsList, verbose=verbose)
  return commentsList

print(reddit.user.me())