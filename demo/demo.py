import praw
import csv
def redditScraper(x):
    reddit = praw.Reddit(user_agent='CommentScraper (by /u/ConnectBad)',
                         client_id='dHUYT7-TgK2aKA', client_secret="Sg2ZreLSzOYqURbJK4QA5Ysh7tM",
                         username='ConnectBad', password='passwordfor180')

    subreddit = reddit.subreddit(x)

    myFile = open('arsenal_reddit_comments.csv', 'w') 
    writer = csv.writer(myFile, dialect='excel')
    writer.writerow(['First 2000 Comments for each post on Hot Page\'s top 40'])

    counter = 0;
    for submission in subreddit.hot(limit=20):
        submission.comments.replace_more(limit=200)
        for comment in submission.comments.list():
            writer.writerow([comment.body.encode('utf-8')])
            counter = counter + 1
            if counter >= 2000:
                break
    myFile.close()
	

import json
import datetime
import csv
import time
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request

def request_until_succeed(url):
    req = Request(url)
    success = False
    while success is False:
        try:
            response = urlopen(req)
            if response.getcode() == 200:
                success = True
        except Exception as e:
            #print(e)
            time.sleep(5)

            #print("Error for URL {}: {}".format(url, datetime.datetime.now()))
            #print("Retrying.")
            return 0

    return response.read()


# Needed to write tricky unicode correctly to csv
def unicode_decode(text):
    try:
        return text.encode('utf-8').decode()
    except UnicodeDecodeError:
        return text.encode('utf-8')


def getFacebookPageFeedUrl(base_url):

    # Construct the URL string; see http://stackoverflow.com/a/37239851 for
    # Reactions parameters
    fields = "&fields=message,link,created_time,type,name,id," + \
        "comments.limit(0).summary(true),shares,reactions" + \
        ".limit(0).summary(true)"

    return base_url + fields


def getFacebookCommentFeedUrl(base_url):

    # Construct the URL string
    fields = "&fields=id,message"
    url = base_url + fields

    return url


def getReactionsForStatuses(base_url):

    reaction_types = ['like', 'love', 'wow', 'haha', 'sad', 'angry']
    reactions_dict = {}   # dict of {status_id: tuple<6>}

    for reaction_type in reaction_types:
        fields = "&fields=reactions.type({}).limit(0).summary(total_count)".format(
            reaction_type.upper())

        url = base_url + fields

        reqResult = request_until_succeed(url)

        if(reqResult != 0):
            data = json.loads(reqResult)['data']

        data_processed = set()  # set() removes rare duplicates in statuses
        for status in data:
            id = status['id']
            count = status['reactions']['summary']['total_count']
            data_processed.add((id, count))

        for id, count in data_processed:
            if id in reactions_dict:
                reactions_dict[id] = reactions_dict[id] + (count,)
            else:
                reactions_dict[id] = (count,)

    return reactions_dict


def processFacebookPageFeedStatus(status):

    # The status is now a Python dictionary, so for top-level items,
    # we can simply call the key.

    # Additionally, some items may not always exist,
    # so must check for existence first

    status_id = status['id']
    status_type = status['type']

    status_message = '' if 'message' not in status else \
        unicode_decode(status['message'])
    link_name = '' if 'name' not in status else \
        unicode_decode(status['name'])
    status_link = '' if 'link' not in status else \
        unicode_decode(status['link'])

    # Time needs special care since a) it's in UTC and
    # b) it's not easy to use in statistical programs.

    status_published = datetime.datetime.strptime(
        status['created_time'], '%Y-%m-%dT%H:%M:%S+0000')
    status_published = status_published + \
        datetime.timedelta(hours=-5)  # EST
    status_published = status_published.strftime(
        '%Y-%m-%d %H:%M:%S')  # best time format for spreadsheet programs

    # Nested items require chaining dictionary keys.

    num_reactions = 0 if 'reactions' not in status else \
        status['reactions']['summary']['total_count']
    num_comments = 0 if 'comments' not in status else \
        status['comments']['summary']['total_count']
    num_shares = 0 if 'shares' not in status else status['shares']['count']

    return (status_id, status_message, link_name, status_type, status_link,
            status_published, num_reactions, num_comments, num_shares)


def scrapeFacebookPageFeedStatus(page_id, access_token, since_date, until_date):
    with open('{}_facebook_statuses.csv'.format(page_id), 'w') as file:
        w = csv.writer(file)
        w.writerow(["status_id", "status_message", "link_name", "status_type",
                    "status_link", "status_published", "num_reactions",
                    "num_comments", "num_shares", "num_likes", "num_loves",
                    "num_wows", "num_hahas", "num_sads", "num_angrys",
                    "num_special"])

        has_next_page = True
        num_processed = 0
        scrape_starttime = datetime.datetime.now()
        after = ''
        base = "https://graph.facebook.com/v2.12"
        node = "/{}/posts".format(page_id)
        parameters = "/?limit={}&access_token={}".format(100, access_token)
        since = "&since={}".format(since_date) if since_date \
            is not '' else ''
        until = "&until={}".format(until_date) if until_date \
            is not '' else ''

        print("Scraping {} Facebook Page: {}\n".format(page_id, scrape_starttime))

        while has_next_page:
            after = '' if after is '' else "&after={}".format(after)
            base_url = base + node + parameters + after + since + until

            url = getFacebookPageFeedUrl(base_url)

            reqResult = request_until_succeed(url)
            
            if(reqResult != 0):
                statuses = json.loads(reqResult)

            reactions = getReactionsForStatuses(base_url)

            for status in statuses['data']:

                # Ensure it is a status with the expected metadata
                if 'reactions' in status:
                    status_data = processFacebookPageFeedStatus(status)
                    reactions_data = reactions[status_data[0]]

                    # calculate thankful/pride through algebra
                    num_special = status_data[6] - sum(reactions_data)
                    w.writerow(status_data + reactions_data + (num_special,))

                num_processed += 1
                if num_processed % 100 == 0:
                    print("{} Statuses Processed: {}".format
                          (num_processed, datetime.datetime.now()))

            # if there is no next page, we're done.
            if 'paging' in statuses:
                after = statuses['paging']['cursors']['after']
            else:
                has_next_page = False

        print("\nDone!\n{} Statuses Processed in {}".format(
              num_processed, datetime.datetime.now() - scrape_starttime))


def processFacebookComment(comment, status_id, parent_id=''):

    # The status is now a Python dictionary, so for top-level items,
    # we can simply call the key.

    # Additionally, some items may not always exist,
    # so must check for existence first

    #comment_id = comment['id']
    comment_message = '' if 'message' not in comment or comment['message'] \
        is '' else unicode_decode(comment['message'])

    # Return a tuple of all processed data

    #return (comment_id, status_id, parent_id, comment_message)
    return(comment_message, '')


def scrapeFacebookPageFeedComments(page_id, access_token, file_id):
    with open('{}_facebook_comments.csv'.format(file_id), 'w') as file:
        w = csv.writer(file)
        w.writerow(["message"])

        num_processed = 0
        scrape_starttime = datetime.datetime.now()
        after = ''
        base = "https://graph.facebook.com/v2.12"
        parameters = "/?limit={}&access_token={}".format(
            100, access_token)

        print("Scraping {} Comments From Posts: {}\n".format(
            file_id, scrape_starttime))

        with open('{}_facebook_statuses.csv'.format(file_id), 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            # Uncomment below line to scrape comments for a specific status_id
            # reader = [dict(status_id='5550296508_10154352768246509')]

            for status in reader:
                has_next_page = True

                while has_next_page:

                    node = "/{}/comments".format(status['status_id'])
                    after = '' if after is '' else "&after={}".format(after)
                    base_url = base + node + parameters + after

                    # DEBUG: BASE URL
                    # print(base_url)

                    url = getFacebookCommentFeedUrl(base_url)
                    
                    # DEBUG: URL
                    # print(url)
                    
                    reqResult = request_until_succeed(url)

                    if(reqResult != 0):
                        comments = json.loads(reqResult)

                    for comment in comments['data']:
                        comment_data = processFacebookComment(
                            comment, status['status_id'])

                        # calculate thankful/pride through algebra
                        if(len(comment_data[0]) > 0):
                            w.writerow(comment_data)

                        if 'comments' in comment:
                            has_next_subpage = True
                            sub_after = ''

                            while has_next_subpage:
                                sub_node = "/{}/comments".format(comment['id'])
                                sub_after = '' if sub_after is '' else "&after={}".format(
                                    sub_after)
                                sub_base_url = base + sub_node + parameters + sub_after

                                sub_url = getFacebookCommentFeedUrl(
                                    sub_base_url)

                                reqResult = request_until_succeed(sub_url)
                                
                                if(reqResult != 0):
                                    sub_comments = json.loads(reqResult)
                                

                                for sub_comment in sub_comments['data']:
                                    sub_comment_data = processFacebookComment(
                                        sub_comment, status['status_id'], comment['id'])

                                    if(len(sub_comment_data[0]) >0):
                                        w.writerow(sub_comment_data)

                                    num_processed += 1
                                    if num_processed % 100 == 0:
                                        print("{} Comments Processed: {}".format(
                                            num_processed,
                                            datetime.datetime.now()))

                                if 'paging' in sub_comments:
                                    if 'next' in sub_comments['paging']:
                                        sub_after = sub_comments[
                                            'paging']['cursors']['after']
                                    else:
                                        has_next_subpage = False
                                else:
                                    has_next_subpage = False

                        # output progress occasionally to make sure code is not
                        # stalling
                        num_processed += 1
                        if num_processed % 1000 == 0:
                            print("{} Comments from facebook Processed: {}".format(
                                num_processed, datetime.datetime.now()))
                        if num_processed >= 2000:
                            break
                    if 'paging' in comments:
                        if 'next' in comments['paging']:
                            after = comments['paging']['cursors']['after']
                        else:
                            has_next_page = False
                    else:
                        has_next_page = False

        print("\nDone!\n")

def facebookscraper(page_id):
    
    app_id = "155822221799033"
    app_secret = "9f827ca98de88187071454a5fcceaa80"
    page_id = "arsenal"
    file_id = page_id                           # set file_id same as page_id

    # input date formatted as YYYY-MM-DD
    since_date = "2017-12-01"
    until_date = ""

    access_token = app_id + "|" + app_secret
    scrapeFacebookPageFeedStatus(page_id, access_token, since_date, until_date)
    scrapeFacebookPageFeedComments(file_id, access_token,file_id)
    print("done")
	
	
	
from pprint import pprint
import nltk
import yaml
import sys
import os
import re
import string

class Normalizer(object):
    def __init__(self):
        pass
    
    def normalize(self, text_input):
        text = text_input.split(' ')
        text = [x.lower() for x in text]
        text = [x.replace("\\n"," ") for x in text ]        
        text = [x.replace("\\t"," ") for x in text ]        
        text = [x.replace("\\xa0"," ") for x in text ]
        text = [x.replace("\\xc2"," ") for x in text ]

        #text = [x.replace(","," ").replace("."," ").replace(" ", "  ") for x in text ]
        #text = [re.subn(" ([a-z]) ","\\1", x)[0] for x in text ]  
        #text = [x.replace("  "," ") for x in text ]

        text = [x.replace(" u "," you ") for x in text ]
        text = [x.replace(" em "," them ") for x in text ]
        text = [x.replace(" da "," the ") for x in text ]
        text = [x.replace(" yo "," you ") for x in text ]
        text = [x.replace(" ur "," you ") for x in text ]
        #text = [x.replace(" ur "," your ") for x in text ]
        #text = [x.replace(" ur "," you're ") for x in text ]

        text = [x.replace("won't", "will not") for x in text ]
        text = [x.replace("can't", "cannot") for x in text ]
        text = [x.replace("i'm", "i am") for x in text ]
        text = [x.replace(" im ", " i am ") for x in text ]
        text = [x.replace("ain't", "is not") for x in text ]
        text = [x.replace("'ll", " will") for x in text ]
        text = [x.replace("'t", " not") for x in text ]
        text = [x.replace("'ve", " have") for x in text ]
        text = [x.replace("'s", " is") for x in text ]
        text = [x.replace("'re", " are") for x in text ]
        text = [x.replace("'d", " would") for x in text ]
        
        punctuation = set(string.punctuation)
        text_new = []
        for word in text:
            word = ''.join([c for c in word.lower() if not c in punctuation])
            text_new.append(word);
        string_new = " ".join(text_new)
        return string_new
    
class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences

class POSTagger(object):

    def __init__(self):
        pass
        
    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]

        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos

class DictionaryTagger(object):

    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]        
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0

def sentence_score(sentence_tokens, previous_token, acum_score):    
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):
    return sum([sentence_score(sentence, None, 0.0) for sentence in review])

def sentiment_anlysis(text):
    
    normalizer = Normalizer()    
    splitter = Splitter()
    postagger = POSTagger()
    dicttagger = DictionaryTagger([ 'dicts/positive.yml', 'dicts/negative.yml', 
                                    'dicts/inc.yml', 'dicts/dec.yml', 'dicts/inv.yml'])

    normalized_text = normalizer.normalize(text)
    
    splitted_sentences = splitter.split(normalized_text)
    
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)

    score = sentiment_score(dict_tagged_sentences)
    length = 1
    if score != 0:
        lengthlist = []
        for n in splitted_sentences:
            lengthlist = lengthlist + n
        length = len(lengthlist)    
    return (score / (length))


def plot_result(x):
    import matplotlib.pyplot as plt
    plt.hist(x)
    plt.title('Histogram of score distribution')
    plt.show()

    
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentimentResult(inputfilename):
    filename = inputfilename.split('.')
    outputfile1 = filename[0] + '_overall_score.csv'  
    outputfile2 = filename[0] + '_all_score.csv'
    analyzer = SentimentIntensityAnalyzer()
    with open(outputfile1, 'wb') as csvfile:
        spamwriter1 = csv.DictWriter(csvfile,fieldnames=['score'])
        spamwriter1.writeheader()
        with open(outputfile2, 'wb') as csvfile:
            spamwriter = csv.DictWriter(csvfile,fieldnames=['Message', 'Abusive_Score', 'NLTK_Neg', 'NLTK_Neu','NLTK_Pos', 'NLTK_Compound', 'Overall_score'])
            spamwriter.writeheader()
            score_sheet = []
            with open(inputfilename, 'rb') as csvfile:
                print 'Start doing sentiments analysis with file',inputfilename
                spamreader = csv.reader(csvfile, delimiter='\t')
                i = 0;
                for row in spamreader:
                    i += 1
                    ## stack overflow for recursion
                    try:
                        line = unicode(row[0][0:min(900,len(row[0]))], 'utf-8').lower()                    
                    except UnicodeDecodeError:
                        continue
                    except IndexError:
                        continue
                    score = sentiment_anlysis(line)

                    if i % 500 == 0: 
                        print 'sentiment analysis done at index', i  
                    result = analyzer.polarity_scores(line)
                    #print(result)
                    if(score != 0):
                        overall_score = max(min(score/(score*score+15)**(1/2.0)*10,-1*result['neg']),-1)
                    else:
                        overall_score = result['compound']
                    score_sheet.append(overall_score)
                    spamwriter1.writerow({'score':overall_score})
                    my_result = {'Message': row[0], 'Abusive_Score': score, 'NLTK_Neg': result['neg'], 'NLTK_Neu': result['neu'], 'NLTK_Pos': result['pos'], 'NLTK_Compound': result['compound'],'Overall_score' : overall_score}
                    if(score !=0):
                        spamwriter.writerow(my_result)
    return score_sheet

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
def plot(x, titleName):
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

def myPlot(dirname, titleName):
    fileDir = dirname
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

if __name__ == '__main__':
    facebookscraper('arsenal')
    print('2000 facebook Comments for arsenal')
    redditScraper('gunners')
    print('2000 Reddit Comments for arsenal')
    facebook_comment_filename = 'arsenal_facebook_comments.csv'
    reddit_comment_filename ='arsenal_reddit_comments.csv'
    
    sentimentResult('arsenal_facebook_comments.csv')
    directory_facebook = 'facebook'
    if not os.path.exists(directory_facebook):
        os.makedirs(directory_facebook)
    os.rename("arsenal_facebook_overall_score.csv", "./facebook")
    print('arsenal facebook comments analysis finished')
    
    sentimentResult('arsenal_reddit_comments.csv')
    directory_reddit = 'reddit'
    if not os.path.exists(directory_reddit):
        os.makedirs(directory_reddit)
    os.rename("arsenal_reddit_overall_score.csv", "./reddit")
    print('arsenal reddit comments analysis finished')
    
    myPlot('./facebook', 'Arsenal_demo_facebook')
    myPlot('./reddit', 'Arsenal_demo_reddit')
    
    os.rename("arsenal_reddit_all_score.csv", "./reddit")
    os.rename("arsenal_facebook_all_score.csv", "./facebook")