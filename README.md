# compare-toxicity

ECE 180 Project by Group 1.

## Puropose
In the real world, people treat others with politeness and hospitality, but in the virtual world people tend to speak without considering the courtesy and morality of the situation. We want to analyze how conflicting people’s overall reactions would be to a certain topic when their anonymity is taken away. The importance of this project is to understand how anonymity affects people’s behavior on the internet.

The two social media platforms we'll compare are Reddit and Facebook.

## Getting Started

**Clone the repository:**
```sh
$ git clone https://github.com/csush/compare-toxicity
```

**Installing Required Packages:**

Install praw package: The Python Reddit API Wrapper
```sh
$ sudo pip install praw
```

Install nltk package for sentiment analysis:
```sh
$ sudo pip install nltk
```
 
*You can also installing packages in Anaconda*
 

**Running Demo:**

Demo introduction: 
Demo aotumatically extracts newest 2000 comments from Reddit and 2000 comments under the topic of Arsenal in facebook and gunners in Reddit. Then the sentiments analysis model will kick in to achieve data preprocessing and analysis for each set of comments. Finally the pie chat and histogram of these two datasets will be plotted.

*recommended environment: Python 2.7 qtconsole*
```sh
$ run demo.py
```
