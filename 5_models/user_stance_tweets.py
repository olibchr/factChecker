import datetime
import json, glob, time, random
import sys, os
import numpy as np
from dateutil import parser
import pickle
sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
sys.path.insert(0, os.path.dirname(__file__) +  '../../991_packages/liblinear-2.20/python/')
from User import User
from Transaction import Transaction
from Fact import Fact
from liblinearutil import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

SERVER_RUN = False

DIR = os.path.dirname(__file__) + '../../3_Data/'

query_dir = DIR + "user_tweet_query_mod/*_search_results.csv"
snippets_dir = DIR + "user_snippets/"

def user_decoder(obj):
    if 'user_id' not in obj.keys(): return obj
    # <user_id, tweets, fact, transactions, credibility, controversy>
    return User(obj['user_id'], obj['tweets'], obj['fact'], obj['transactions'], obj['credibility'],
                obj['controversy'], obj['features'], obj['was_correct'])


def get_snippets(user):
    snippet_files = glob.glob(snippets_dir)
    this_snippet_file = [snip for snip in snippet_files if str(user.user_id) == snip[snip.rfind('/')+1:snip.rfind('_')]][0]
    snippets = json.loads(open(this_snippet_file).readline())
    return snippets

def get_users():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    print('{} users'.format(len(user_files)))
    if len(user_files) < 10: print('WRONG DIR?')
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=user_decoder)
        yield user

def stance_analysis(user):
    sid = SentimentIntensityAnalyzer()
    user.snippets = get_snippets(user)
    for snip in user.snippets:
        tweet_sent = sid.polarity_scores(snip['query'])
        snip_sents_uni = []
        snip_sents_bi = {}
        for snipsnip in snip['snippets']['unigrams']:
            snip_sents_uni.append(sid.polarity_scores(snipsnip))

users = get_users()

modelPath = "./resources/model_file_filtered_w500_bigram";


