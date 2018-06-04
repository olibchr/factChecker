import sys, json, glob, os, hashlib, pandas
sys.path.insert(0, os.path.dirname(__file__) + '../../2_helpers')
from Transaction import Transaction
from Fact import Fact
from User import User
from dateutil import parser
from collections import Counter
import datetime, hashlib
import numpy as np
import csv
import tweepy

DIR = os.path.dirname(__file__) + '../../../4_RawData/Castillo/'
OUT = os.path.dirname(__file__) + '../../../5_DataCastillo/'

ALT_ACCOUNT = False

# Generate your own at https://apps.twitter.com/app
CONSUMER_KEY = '4Y897wHsZJ2Qud1EgncojnoNS'
CONSUMER_SECRET = 'sMpckIKpf00c1slGciCe4FvWlUTkFUGKkMAu88x2SBdJRW3laR'
OAUTH_TOKEN = '1207416314-pX3roPjOm0xNuGJxxRFfE6H0CyHRCgnzXvNfFII'
OAUTH_TOKEN_SECRET = 'NVS29lZafbCF4kvc1yCEKg0f00AYE3Ogj7XkygHsBI5LD'
if ALT_ACCOUNT:
    CONSUMER_KEY = '0pUhFi92XQbPTB70eEnhJ0fTH'
    CONSUMER_SECRET = 'DLjLTOoonzO5ADVfIppnLmMpCL1qM9fOHkhoXfXYIQXe3hvC9W'
    OAUTH_TOKEN = '978935525464858624-uBlhj4nIUr2eEJghiNkSzFO25hcHW2I'
    OAUTH_TOKEN_SECRET = 'eqgP2jzCzJVqcWxaqwTbFeHWKjDvMEKD6YR78UNhse6qp'

# Objectives:
# Build Users + Facts

def get_facts_tweet():
    tweet_to_fact = {}
    with open(DIR + 'trendid-tweetid.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            tweet_to_fact[r[1]] = r[0]
    print("trendid-tweetid: {}".format(len(tweet_to_fact)))
    return tweet_to_fact

def get_facts_y():
    fact_to_cred = {}
    with open(DIR + 'labels-trendid-credible.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            fact_to_cred[r[0]] = r[1]
    print("Topcis classified with credibility: {}".format(len(fact_to_cred)))
    return fact_to_cred

def get_facts_type():
    fact_to_type = {}
    with open(DIR + 'labels-trendid-newsworthy.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            fact_to_type[r[0]] = r[1]
    print("labels-trendid-newsworthy: {}".format(len(fact_to_type)))
    return fact_to_type


def build_facts(fact_to_cred, fact_to_type, fact_to_user):
    facts = []
    for f in fact_to_cred.items():
        true = 0 if f[1] == 'NOTCREDIBLE' else 1
        type = fact_to_type[f[0]] if f[0] in fact_to_type else 0
        # <RUMOR_TPYE, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND, SOURCE_TWEET, ?HASH>
        this_fact = Fact(type, '', '', true, 0, 0, 0, hash=f[0])
        facts.append(this_fact)
    return facts


def build_transactions(tweet_to_fact, fact_to_cred):
    transactions = []
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)

    for fact, cred in fact_to_cred.items():
        tweets_for_fact = sorted([k for k,v in tweet_to_fact.items() if v == fact])[:100]
        tweets = api.statuses_lookup(tweets_for_fact)

        for status in tweets: # tweepy.Cursor(api.statuses_lookup, id=tweets_for_fact).items():
            tr = Transaction(tweets_for_fact[0], status._json['id_str'], status._json['user']['id_str'], fact, status._json['created_at'], -1, -1, status._json['text'])
            transactions.append(tr)

    return transactions



def store_result():
    with open(OUT + 'facts.json', 'w') as out_file:
        out_file.write(json.dumps([f.__dict__ for f in FACTS]))
    with open(OUT + 'factTransaction.json', 'w') as out_file:
        out_file.write(json.dumps([f.__dict__ for f in TRANSACTIONS]))

fact_to_cred = get_facts_y()
fact_to_type = get_facts_type()
tweet_to_fact = get_facts_tweet()
tweet_to_fact = {k:v for k,v in tweet_to_fact.items() if v in fact_to_cred}
print("Tweets on topics that are classified as Cred or not: {}".format(len(tweet_to_fact)))

FACTS = build_facts(fact_to_cred, fact_to_type, tweet_to_fact)

TRANSACTIONS = build_transactions(tweet_to_fact, fact_to_cred)

store_result()