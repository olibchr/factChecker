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

sys.path.insert(0, os.path.dirname(__file__) + '../../2_helpers')
from decoder import decoder


num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

DIR = os.path.dirname(__file__) + '../../../5_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))

neg_words_f = glob.glob(DIR + 'model_data/negative-words.txt')[0]
pos_words_f = glob.glob(DIR + 'model_data/positive-words.txt')[0]
with open(neg_words_f, 'r', encoding = "ISO-8859-1") as f:
    neg_words = [w.strip().replace('\n','') for w in f.readlines()]
with open(pos_words_f, 'r', encoding = "ISO-8859-1") as f:
    pos_words = [w.strip().replace('\n','') for w in f.readlines()]
print(len(neg_words), neg_words[:10])
sid = SentimentIntensityAnalyzer()


def get_users(user_files):
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
    print(user.user_id)
    user.was_correct = -1
    transaction = transactions[0]
    for tr in transactions:
        if str(tr.user_id) == str(user.user_id):
            transaction = tr
            user.transactions = tr
            transactions.remove(tr)
            user.fact = transaction.fact
            user.fact_text = transaction.text
            user.fact_text_ts = transaction.timestamp
            user.stance = transaction.stance
            break
    if transaction is None: return user
    for fact in facts:
        if fact.hash == transaction.fact:
            user.fact_text = transaction.text
            stance = sid.polarity_scores(user.fact_text)['compound']
            if (str(fact.true) == '1' and stance > 0.5) or (
                            str(fact.true) == '0' and stance < -0.5):
                user.was_correct = 1
            elif(str(fact.true) == '1' and stance < -0.5) or \
                    (str(fact.true) == '0' and stance > 0.5):
                user.was_correct = 0
            break
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
    user.avg_time_to_retweet = 0
    if not user.tweets or len(user.tweets) < 1: return user
    time_btw_rt = []
    for tweet in user.tweets:
        if not 'quoted_status' in tweet: continue
        if not 'created_at' in tweet['quoted_status']: continue
        date_original = parser.parse(tweet['quoted_status']['created_at'])
        date_retweet = parser.parse(tweet['created_at'])
        time_btw_rt.append(date_original - date_retweet)
    if len(time_btw_rt) == 0: return user

    average_timedelta = round(float((sum(time_btw_rt, datetime.timedelta(0)) / len(time_btw_rt)).seconds) / 60)
    user.avg_time_to_retweet = average_timedelta
    return user


def get_data():
    fact_file = glob.glob(DIR + 'facts.json')[0]
    transactions_file = glob.glob(DIR + 'factTransaction.json')[0]
    facts = json.load(open(fact_file), object_hook=decoder)
    transactions = json.load(open(transactions_file), object_hook=decoder)
    transactions = sorted(transactions, reverse=True, key=lambda t: t.user_id)
    return facts, transactions


def store_result(user):
    with open(DIR + 'user_tweets/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(str(json.dumps(user.__dict__, default=datetime_converter)) + '\n')


def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def main():
    wn.ensure_loaded()
    # batch
    i = int(sys.argv[1])
    user_files = sorted(glob.glob(DIR + 'user_tweets/' + 'user_*.json'))
    r = 1960
    user_files = user_files[(i-1)*r:min(r*i,len(user_files))]
    users = get_users(user_files)
    facts, transactions = get_data()
    users = Parallel(n_jobs=num_jobs)(delayed(was_user_correct)(user, facts, transactions) for user in users)
    # print("Linguistic features..")
    # users = Parallel(n_jobs=num_jobs)(delayed(linguistic_f)(user) for user in users)
    # print("Calculating tweet sentiment for each user")
    # users = Parallel(n_jobs=num_jobs)(delayed(feature_user_tweet_sentiment)(user) for user in users)
    # print("Avg time to retweet")
    # users = Parallel(n_jobs=num_jobs)(delayed(time_til_retweet)(user) for user in users)
    # print([u.sent_tweets_avg for u in users[:10]])
    # print([u.avg_time_to_retweet for u in users[:10]])
    [store_result(user) for user in users]


if __name__ == "__main__":
    main()