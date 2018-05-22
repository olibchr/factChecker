import datetime
import glob
import json
import multiprocessing
import os
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from string import digits
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from dateutil import parser
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
from decoder import decoder


num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

DIR = os.path.dirname(__file__) + '../../3_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))

neg_words_f = glob.glob(DIR + 'model_data/negative-words.txt')[0]
pos_words_f = glob.glob(DIR + 'model_data/positive-words.txt')[0]
with open(neg_words_f, 'r', encoding = "ISO-8859-1") as f:
    neg_words = f.readlines()
with open(pos_words_f, 'r', encoding = "ISO-8859-1") as f:
    pos_words = f.readlines()
print(len(neg_words), neg_words[:10])
sid = SentimentIntensityAnalyzer()


def get_users():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    print('{} users'.format(len(user_files)))
    if len(user_files) < 10: print('WRONG DIR?')
    users = []
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=decoder)
        users.append(user)
    return users


def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return [WNL.lemmatize(i.lower()) for i in tokenizer.tokenize(text) if
            i.lower() not in NLTK_STOPWORDS]


def was_user_correct(user, facts, transactions):
    for tr in transactions:
        if str(tr.user_id) == str(user.user_id):
            transaction = tr
            user.transactions = tr
            transactions.remove(tr)
            user.fact = transaction.fact
            user.fact_text = transaction.text
            user.fact_text_ts = transaction.timestamp
            user.stance = 0 if transaction.stance == 'denying' else 1 if transaction.stance == 'supporting' else 2 if transaction.stance == 'comment' else 3
            break
    for fact in facts:
        if fact.hash == transaction.fact:
            user.text = transaction.text
            if (str(fact.true) == '1' and transaction.stance == 'supporting') or (
                            str(fact.true) == '0' and transaction.stance == 'denying'):
                user.was_correct = 1
            elif(str(fact.true) == '1' and transaction.stance == 'denying') or \
                    (str(fact.true) == '0' and transaction.stance == 'supporting'):
                user.was_correct = 0
            else:
                user.was_correct = -1
            print(fact.true, transaction.stance, user.was_correct)
    return user


def linguistic_f(user):
    user_pos_words = 0
    user_neg_words = 0
    if not user.tweets: return user
    for tweet in user.tweets:
        for token in tokenize_text(tweet['text']):
            if token in neg_words:
                user_neg_words += 1
            if token in pos_words:
                user_pos_words += 1
    if user.features is None: user.features = {}; print(user.user_id)
    user.features['pos_words'] = user_pos_words
    user.features['neg_words'] = user_neg_words
    return user

    # bias
    assertives = ['think', 'believe', 'suppose', 'expect', 'imagine']
    factives = ['know', 'realize', 'regret', 'forget', 'find out']
    hedges = ['postulates', 'felt', 'likely', 'mainly', 'guess']
    implicatives = ['manage', 'remember', 'bother', 'get', 'dare']
    # subjectivity
    wiki_Bias_Lexicon = ['apologetic', 'summer', 'advance', 'cornerstone']
    negative = ['hypocricy', 'swindle', 'unacceptable', 'worse']
    positive = ['steadiest', 'enjoyed', 'prominence', 'lucky']
    subj_clues = ['better', 'heckle', 'grisly', 'defeat', 'peevish']
    affective = ['disgust', 'anxious', 'revolt', 'guilt', 'confident']


def feature_user_tweet_sentiment(user):
    if not user.tweets: return user
    tweet_sents = []
    for tweet in user.tweets:
        ss = sid.polarity_scores(tweet['text'])
        tweet_sents.append(ss['compound'])
    #density, _ = np.histogram(tweet_sents, bins=bins, density=True)
    #user.sent_tweets_density = density / density.sum()
    user.sent_tweets_avg = np.average(tweet_sents)
    return user


def time_til_retweet(user):
    print("Calculating avg time between original tweet and retweet per user")
    if not user.tweets or len(user.tweets) < 1: return user
    time_btw_rt = []
    if user.avg_time_to_retweet is None:
        for tweet in user.tweets:
            if not 'quoted_status' in tweet: return user
            if not 'created_at' in tweet['quoted_status']: return user
            date_original = parser.parse(tweet['quoted_status']['created_at'])
            date_retweet = parser.parse(tweet['created_at'])
            time_btw_rt.append(date_original - date_retweet)
        if len(time_btw_rt) == 0: return user

        average_timedelta = round(float((sum(time_btw_rt, datetime.timedelta(0)) / len(time_btw_rt)).seconds) / 60)
        user.avg_time_to_retweet = average_timedelta
    return user


def store_result(user):
    with open(DIR + 'user_tweets/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')


def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


def main():
    wn.ensure_loaded()
    users = get_users()

    #users = [was_user_correct(user) for user in users]
    print("Linguistic features..")
    users = Parallel(n_jobs=num_jobs)(delayed(linguistic_f)(user) for user in users)
    #print("Calculating tweet sentiment for each user")
    #users = Parallel(n_jobs=num_jobs)(delayed(feature_user_tweet_sentiment)(user) for user in users)
    #print("Avg time to retweet")
    #users = Parallel(n_jobs=num_jobs)(delayed(time_til_retweet)(user) for user in users)
    [store_result(user) for user in users]


if __name__ == "__main__":
    main()