from __future__ import print_function
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
import re

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
from gensim.models import KeyedVectors
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn import metrics

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
sys.path.insert(0, os.path.dirname(__file__) + '../5_models')
from decoder import decoder
from metrics import ndcg_score

warnings.filterwarnings("ignore", category=DeprecationWarning)

DIR = os.path.dirname(__file__) + '../../3_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))

num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

fact_to_words = {}
#word_vectors = KeyedVectors.load_word2vec_format('model_data/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')

def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def tokenize_text(text, only_retweets=False):
    tokenizer = RegexpTokenizer(r'\w+')
    if only_retweets:
        text = text
        #if 'RT' not in text: return None
        mentions = []
        while True:
            if '@' not in text: break
            mention = text[text.find('@'):]
            if ' ' in mention: mention = mention[:mention.find(' ')]
            mentions.append(mention)
            text = text.replace(mention,'')
        retweeted_to = [rt.replace('@','').replace(':','').lower() for rt in mentions if '@' in rt]
        return retweeted_to
    return [WNL.lemmatize(i.lower()) for i in tokenizer.tokenize(text) if
            i.lower() not in NLTK_STOPWORDS]


def get_data():
    fact_file = glob.glob(DIR + 'facts.json')[0]
    transactions_file = glob.glob(DIR + 'factTransaction.json')[0]
    facts = json.load(open(fact_file), object_hook=decoder)
    transactions = json.load(open(transactions_file), object_hook=decoder)
    return facts, transactions


def get_users():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    print('{} users'.format(len(user_files)))
    if len(user_files) < 10: print('WRONG DIR?')
    users = []
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=decoder)
        if int(user.was_correct) != -1:
            users.append(user)
    print('Kept {} users'.format(len(users)))
    return users


def get_relevant_tweets(user):
    relevant_tweets= []
    user_fact_words = fact_to_words[user.fact]
    for tweet in user.tweets:
        distance_to_topic = []
        tokens = tokenize_text(tweet['text'], only_retweets=False)
        for token in tokens:
            if token not in word_vectors.vocab: continue
            increment = np.average(word_vectors.distances(token, other_words=user_fact_words))
            distance_to_topic.append(increment)
        if np.average(np.asarray(distance_to_topic)) < 0.8:
            relevant_tweets.append(tweet)

    return relevant_tweets


def build_fact_topics():
    print("Build fact topics")
    fact_file = glob.glob(DIR + 'facts_annotated.json')[0]
    facts_df = pd.read_json(fact_file)
    remove_digits = str.maketrans('', '', digits)
    facts_df['text_parsed'] = facts_df['text'].map(lambda t: tokenize_text(t.translate(remove_digits)))
    facts_df['entities_parsed'] = facts_df['entities'].map(lambda ents:
                                                           [item for sublist in
                                                            [e['surfaceForm'].lower().split() for e in ents if
                                                             e['similarityScore'] >= 0.6]
                                                            for item in sublist])
    facts_df['topic'] = facts_df['topic'].map(lambda t: [t])
    facts_df['fact_terms'] = facts_df['text_parsed'] + facts_df['entities_parsed'] + facts_df['topic']
    return facts_df


def get_user_edges(users):
    user_to_links = []
    y = []
    i = 0
    for user in users:
        user_links = []
        #relevant_tweets = get_relevant_tweets(user)
        for tweet in user.tweets:
            mentions = tokenize_text(tweet['text'], only_retweets=True)
            for rt in mentions:
                user_links.append(rt)
        if len(user_links) <= 1: continue
        user_to_links.append([user.user_id, user_links])
        y.append([user.user_id, user.was_correct])
        i += 1
    return user_to_links, np.asarray(y)


def build_graph(user_to_links, user_to_weight):
    G=nx.Graph()
    all_nodes = [u[0] for u in user_to_links] + list(set([e for sublist in [u[1] for u in user_to_links] for e in sublist]))
    G.add_nodes_from(all_nodes)
    #G.add_edges_from([(userlinks[0],v) for userlinks in user_to_links for v in userlinks[1]])
    G.add_weighted_edges_from([(userlinks[0],v,user_to_weight[k]) for userlinks in user_to_links for v in userlinks[1]])
    obsolete_nodes = [k for k,v in dict(nx.degree(G)).items() if v <= 1]
    G.remove_nodes_from(obsolete_nodes)
    return G


def get_ranks(user_to_links, G, pageRank, alpha=0.85):
    user_to_pr = []
    for user, links in user_to_links:
        pr_sum = sum([pageRank[l]/G.degree(l) for l in links if l in pageRank])
        pr_user = (1-alpha)/alpha + alpha*pr_sum
        user_to_pr.append(pr_user)
    return user_to_pr


def rank_users(users):
    global fact_to_words
    print("Creating nodes")
    #fact_topics = build_fact_topics()
    #fact_to_words = {r['hash']: [w for w in r['fact_terms'] if w in word_vectors.vocab] for index, r in fact_topics[['hash', 'fact_terms']].iterrows()}
    user_to_links, user_to_weight = get_user_edges(users)

    X_train, X_test, y_train, y_test = train_test_split(user_to_links, user_to_weight)
    print("Building graph..")
    G = build_graph(X_train, y_train)
    pr = nx.pagerank(G)

    pr_cred_users = {u:v for u,v in list(pr.items()) if u in user_to_links}
    # print(sorted([(v,y[1]) for u,v in pr_cred_users.items() for y in user_to_weight if u == y[0]], reverse=True, key=lambda x: x[0]))

    pred = get_ranks(X_test, G, pr)
    print(sorted(np.asarray([e for e in zip(pred, [y[1] for y in y_test])]), reverse=True, key=lambda x: x[0]))

    ndgc = ndcg_score([y[1] for y in y_test], pred)
    print("NDCG: {}".format(ndgc))



users = get_users()
rank_users(users)

