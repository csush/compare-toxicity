
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
        #adapt format
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
    ##pprint(normalized_text)
    
    splitted_sentences = splitter.split(normalized_text)
    ## pprint(splitted_sentences)
    
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
    ##pprint(pos_tagged_sentences)

    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    ##pprint(dict_tagged_sentences)

    #print("analyzing sentiment...")
    score = sentiment_score(dict_tagged_sentences)
    length = 1
    if score != 0:
        lengthlist = []
        for n in splitted_sentences:
            lengthlist = lengthlist + n
        length = len(lengthlist)    
    return (score / (length))
    #print(score / len(splitted_sentences))
    #print(score)

    
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


with open('Analyzed_result.csv', 'wb') as csvfile:
    spamwriter = csv.DictWriter(csvfile,fieldnames=['Message', 'Score', 'NLTK_Neg', 'NLTK_Neu','NLTK_Pos', 'NLTK_Compound'])
    spamwriter.writeheader()
    
    with open('another_reddit_comments.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        i = 0;
        for row in spamreader:
            i += 1
            ## stack overflow for recursion
            try:
                line = unicode(row[0][0:min(900,len(row[0])], 'utf-8').lower()
            except IndexError:
                continue

            score = sentiment_anlysis(line)
            if score >= 0:
                continue
            print 'index at ', i
            print  row[0]
            print score    
            result = analyzer.polarity_scores(line)
            print(result)
            print '-----------'
            
            my_result = {'Message': row[0], 'Score': score, 'NLTK_Neg': result['neg'], 'NLTK_Neu': result['neu'], 'NLTK_Pos': result['pos'], 'NLTK_Compound': result['compound']}
            spamwriter.writerow(my_result)
