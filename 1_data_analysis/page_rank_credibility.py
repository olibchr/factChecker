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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
from decoder import decoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

DIR = os.path.dirname(__file__) + '../../3_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))

num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)


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


def get_user_edges(users, cred=None):
    user_to_links = {}
    user_to_weight = defaultdict(lambda x: 0.5)
    for user in users:
        if cred != None:
            if user.was_correct != cred: continue
        user_to_links[user.user_id] = []
        for tweet in user.tweets:
            mentions = tokenize_text(tweet['text'], only_retweets=True)
            [user_to_links[user.user_id].append(rt) for rt in mentions if len(mentions) > 0]
        if len(user_to_links[user.user_id]) <= 1: del user_to_links[user.user_id]
        user_to_weight[user.user_id] = user.was_correct
    return user_to_links, user_to_weight


def build_graph(user_to_links):
    G=nx.Graph()
    all_nodes = list(user_to_links.keys()) + list(set([e for sublist in list(user_to_links.values()) for e in sublist]))
    G.add_nodes_from(all_nodes)
    G.add_edges_from([(k,v) for k, sublist_v in user_to_links.items() for v in sublist_v])
    #G.add_weighted_edges_from([(k,v,user_to_weight[k]) for k, sublist_v in user_to_links.items() for v in sublist_v])
    obsolete_nodes = [k for k,v in dict(nx.degree(G)).items() if v <= 1]
    G.remove_nodes_from(obsolete_nodes)
    return G


def get_ranks(user_to_links, G, pageRank, alpha=0.85):
    user_to_pr = {}
    for user, links in user_to_links.items():
        pr_sum = sum([pageRank[l]/G.degree(l) for l in links if l in pageRank])
        pr_user = (1-alpha)/alpha + alpha*pr_sum
        user_to_pr[user] = pr_user
    return user_to_pr


def rank_users(users):
    user_to_links, user_to_weight = get_user_edges(users)
    user_to_weight = {}

    G = build_graph(user_to_links)

    for node in list(G.nodes()):
        if node not in user_to_links: user_to_weight[node] = 0.5
        else: user_to_weight[node] = user_to_weight[node]

    pr = nx.pagerank(G, nstart=user_to_weight)

    pr_cred_users = {u:v for u,v in list(pr.items()) if u in user_to_links}

    pred_cred = get_ranks(user_to_links, G, pr)

    merged_list = sorted(list(pr_cred_users.items()) + list(pred_cred.items()), reverse=True, key=lambda x:x[1])
    print([(r, user_to_weight[r[0]]) for r in merged_list])




users = get_users()
rank_users(users)

