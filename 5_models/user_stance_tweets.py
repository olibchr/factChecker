import datetime
import json, glob, time, random
import sys, os, csv
import numpy as np
from dateutil import parser
import pickle
sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
sys.path.insert(0, os.path.dirname(__file__) +  '../../991_packages/liblinear-2.20/python/')
from User import User
from Transaction import Transaction
from Fact import Fact
from liblinearutil import *
import nltk, re
from nltk import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

SERVER_RUN = False

DIR = os.path.dirname(__file__) + '../../3_Data/'

query_dir = DIR + "user_tweet_query_mod/*_search_results.csv"
snippets_dir = DIR + "user_snippets/"

words_to_idx = {}
idx_to_words = {}

def user_decoder(obj):
    if 'user_id' not in obj.keys(): return obj
    # <user_id, tweets, fact, transactions, credibility, controversy>
    return User(obj['user_id'], obj['tweets'], obj['fact'], obj['transactions'], obj['credibility'],
                obj['controversy'], obj['features'], obj['was_correct'])


def get_snippets(user):
    snippet_files = glob.glob(snippets_dir + '0*_snippets.json')
    print("get snippet")
    print(snippet_files)
    return json.loads(open(snippet_files[0]).readline())
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

def get_word_map():
    global words_to_idx, idx_to_words
    words_to_idx = {}
    idx_to_words = {}
    with open(DIR + '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/2_Code/5_models/wordMap_Bigram.tsv') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader.readline():
            words_to_idx[line[0]] = line[1]
            idx_to_words[line[1]] = line[0]
    return words_to_idx, idx_to_words


def sentiment_analysis(user):
    sid = SentimentIntensityAnalyzer()
    user.snippets = get_snippets(user)
    user_scores = []
    for snip in user.snippets:
        tweet_sent = sid.polarity_scores(snip['query'])
        snip_sents_uni = []
        snip_sents_bi = []
        for snipsnip in snip['snippets']['unigrams']:
            snip_sents_uni.append(sid.polarity_scores(snipsnip))
        for snipsnip in snip['snippets']['bigrams']:
            snip_sents_bi.append(sid.polarity_scores(snipsnip))
        user_scores.append({
            'tweet': tweet_sent,
            'unigrams': snip_sents_uni,
            'bigrams': snip_sents_bi
        })
    return user_scores


def get_features(snippet):
    global words_to_idx, idx_to_words
    feature_map = {}
    tokenized = word_tokenize(snippet)
    for i in range(len(tokenized)-1):
        word = tokenized[i]
        next_word = tokenized[i+1]
        if len(word) <2: continue
        if word not in words_to_idx: continue
        if words_to_idx[word] not in feature_map:
            feature_map[words_to_idx[word]] = 1
        else: feature_map[words_to_idx[word]] += 1

        bigram = word + '_' + next_word
        if bigram not in words_to_idx: continue
        if words_to_idx[bigram] not in feature_map:
            feature_map[words_to_idx[bigram]] = 1
        else: feature_map[words_to_idx[bigram]] += 1
    return feature_map


def stance_analysis(user, model):
    user_stances = []
    user.snippets = get_snippets(user)
    for snip in user.snippets:
        tweet_text = re.sub(r'[^a-z0-9]', ' ', snip['query'].lower())
        tweet_f = get_features(tweet_text)
        print(tweet_f)
        tweet_stance = predict([], tweet_f, model)

        features_uni = []
        features_bi = []
        for snipsnip in snip['snippets']['unigrams']:
            features_uni.append(get_features(snipsnip))
        for snipsnip in snip['snippets']['bigrams']:
            features_bi.append(get_features(snipsnip))
        uni_stances = predict([], problem(features_uni), model)
        bi_stances = predict([], problem(features_bi), model)
        user_stances.append({
            'tweet': tweet_stance,
            'unigrams': uni_stances,
            'bigrams': bi_stances
        })
    return user_stances


modelPath = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/2_Code/5_models/model_file_filtered_w500_bigram'
m = load_model(modelPath)

#for user in get_users():
    #print(sentiment_analysis(user))

for user in get_users():
    print(user.user_id)
    print(stance_analysis(user, m))


