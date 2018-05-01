import glob, os, sys, json, datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import re, nltk
from dateutil import parser
from nltk import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from Fact import Fact
from User import User
from Transaction import Transaction
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from scipy.stats import norm
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)

SERVER_RUN = True

DIR = os.path.dirname(__file__) + '../../3_Data/'


def decoder(o):
    def user_decoder(obj):
        if 'user_id' not in obj.keys(): return obj
        # if 'avg_time_to_retweet' in obj.keys():
        return User(obj['user_id'], tweets=obj['tweets'], fact=obj['fact'], transactions=obj['transactions'],
                    credibility=obj['credibility'],
                    controversy=obj['controversy'], features=obj['features'], was_correct=obj['was_correct'],
                    avg_time_to_retweet=obj['avg_time_to_retweet'] if 'avg_time_to_retweet' in obj.keys() else None,
                    sent_tweets_density=obj['sent_tweets_density'] if 'sent_tweets_density' in obj.keys() else None,
                    sent_tweets_avg=obj['sent_tweets_avg'] if 'sent_tweets_avg' in obj.keys() else None
                    )

    def fact_decoder(obj):
        # <RUMOR_TPYE, HASH, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND, SOURCE_TWEET>
        return Fact(obj['rumor_type'], obj['topic'], obj['text'], obj['true'], obj['proven_false'],
                    obj['is_turnaround'], obj['source_tweet'], hash=obj['hash'])

    def transaction_decoder(obj):
        # <sourceId, id, user_id, fact, timestamp, stance, weight>
        return Transaction(obj['sourceId'], obj['id'], obj['user_id'], obj['fact'], obj['timestamp'], obj['stance'],
                           obj['weight'])

    if 'tweets' in o.keys():
        return user_decoder(o)
    elif 'hash' in o.keys():
        return fact_decoder(o)
    elif 'sourceId' in o.keys():
        return transaction_decoder(o)
    else:
        return o


def datetime_converter(o):
    if isinstance(o, type(datetime)):
        return o.__str__()


def get_data():
    fact_file = glob.glob(DIR + 'facts.json')[0]
    transactions_file = glob.glob(DIR + 'factTransaction.json')[0]
    facts = json.load(open(fact_file), object_hook=decoder)
    transactions = json.load(open(transactions_file), object_hook=decoder)
    return facts, transactions


def get_users():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    print('Found {} users'.format(len(user_files)))
    if SERVER_RUN:
        user_files = sorted(user_files, reverse=False)
    else:
        user_files = sorted(user_files, reverse=True)
    if len(user_files) < 10: print('WRONG DIR?')
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=decoder)
        yield user

def get_web_doc(user):
    doc_dir = DIR + 'user_docs/'
    doc_file = [f for f in glob.glob(doc_dir) if str(user.user_id) in f][0]
    with open(doc_file, 'w') as f:
        web_docs_df = pd.read_json(f)
    return web_docs_df


def feature_user_tweet_sentiment(users):
    sid = SentimentIntensityAnalyzer()
    bins = np.arange(-1, 1.1, 0.2)

    i =0
    for user in users:
        if not user.tweets: continue
        tweet_sents = []
        for tweet in user.tweets:
            ss = sid.polarity_scores(tweet['text'])
            tweet_sents.append(ss['compound'])
        density, _ = np.histogram(tweet_sents, bins=bins, density=True)
        user.sent_tweets_density = density / density.sum()
        user.sent_tweets_avg = np.average(tweet_sents)
        write_user(user)
        i += 1
        if i %100 == 0: print(i)

def feature_user_web_doc_sentiment(users):
    for user in users:
        web_docs_df = get_web_doc(user)
        print(web_docs_df.describe())
        exit()


def write_user(user):
    print("Writing user: {}".format(user.user_id))
    with open(DIR + 'user_tweets/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')


def match_transaction_data(users, transactions):
    for user in users:
        tr = [tr for tr in transactions if tr.user_id == user.user_id][0]
        user.stance = tr.stance
        user.certainty = tr.certainty


# <user_id, tweets, fact, transactions, credibility, controversy, features, was_correct, snippets, avg_time_to_retweet>
# tweets <text, created_at, reply_to, retweets, favorites, *quoted_status<created_at, text>>
def main():
    #facts, df_transactions = get_data()
    #feature_user_tweet_sentiment(get_users())
    feature_user_web_doc_sentiment(get_users())

if __name__ == "__main__":
    main()
