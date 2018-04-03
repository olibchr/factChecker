from datetime import datetime
import json, glob, time, random
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import re, nltk
from User import User
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

NEW_CORPUS = True

DIR = os.path.dirname(__file__) + '../../3_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))


def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


def user_decoder(obj):
    if 'user_id' not in obj.keys(): return obj
    # <user_id, tweets, fact, transactions, credibility, controversy>
    return User(obj['user_id'], obj['tweets'], obj['fact'], obj['transactions'], obj['credibility'],
                obj['controversy'])


def get_users():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    print('Getting Search Results for {} users'.format(len(user_files)))
    if len(user_files) < 10: print('WRONG DIR?')
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=user_decoder)
        yield user

def get_corpus():
    corpus_file = glob.glob('bow_corpus.json')[0]
    bow_corpus = json.loads(open(corpus_file).readline())
    return bow_corpus

def build_bow_corpus(users):
    print("Building a new Bow corpus")
    bow_corpus = {}
    for user in users:
        for tweet in user.tweets:
            # Tweets <text, created_at, *quoted_status
            tokens = [WNL.lemmatize(i) for i in nltk.word_tokenize(tweet['text']) if i not in NLTK_STOPWORDS]
            for token in tokens:
                if token in bow_corpus:
                    bow_corpus[token] += 1
                else:
                    bow_corpus[token] = 1
    return bow_corpus


def corpus_analysis(bow_corpus):
    print("Top occuring terms: {}".format(sorted(bow_corpus.items(), key=lambda w: w[1])[:20]))


def save_corpus(bow_corpus):
    with open('bow_corpus.json', 'w') as out_file:
        out_file.write(json.dumps(bow_corpus, default=datetime_converter) + '\n')

def main():
    global bow_corpus
    wn.ensure_loaded()
    users = get_users()
    if NEW_CORPUS:
        bow_corpus = build_bow_corpus(users)
    else: bow_corpus = get_corpus()
    corpus_analysis(bow_corpus)
    save_corpus(bow_corpus)


if __name__ == "__main__":
    main()
