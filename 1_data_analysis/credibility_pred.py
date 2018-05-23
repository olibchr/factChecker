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
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from scipy.sparse import lil_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
from decoder import decoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

NEW_CORPUS = False
BUILD_NEW_SPARSE = False

DIR = os.path.dirname(__file__) + '../../3_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))

num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

# global Vars
word_to_idx = {}
fact_to_words = {}
#if BUILD_NEW_SPARSE:
word_vectors = KeyedVectors.load_word2vec_format('model_data/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True)


def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def tokenize_text(text, only_retweets=False):
    tokenizer = RegexpTokenizer(r'\w+')
    links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for l in links: text = text.replace(l,' evidence ')
    if only_retweets:
        text = text.lower()
        if 'rt' not in text: return []
        text = text[text.find('rt'):]
        text = text[text.find('@'):text.find(':')]
        return [WNL.lemmatize(i.lower()) for i in tokenizer.tokenize(text) if
                i.lower() not in NLTK_STOPWORDS]
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
        users.append(user)
    return users


def get_corpus():
    corpus_file = glob.glob('model_data/bow_corpus.json')[0]
    bow_corpus = json.loads(open(corpus_file).readline())
    return bow_corpus


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


def get_series_from_user(user):
    relevant_tweets= []
    relevant_tweet_vecs = []
    all_distances = []
    user_fact_words = fact_to_words[user.fact]
    for tweet in user.tweets:
        tokens = tokenize_text(tweet['text'], only_retweets=False)
        # print(tokens, user_fact_words)
        distance_to_topic = []
        for token in tokens:
            if token not in word_to_idx: continue
            if token not in word_vectors.vocab: continue
            increment = np.average(word_vectors.distances(token, other_words=user_fact_words))
            if increment > 1: increment = 1
            if increment < 0: increment = 0
            distance_to_topic.append(increment)
        distance_to_topic = np.average(distance_to_topic)
        all_distances.append(distance_to_topic)
        if distance_to_topic < 0.5:
            relevant_tweets.append(tweet)
            tweet_vec = [word_to_idx[t] for t in tokens if t in word_to_idx]
            relevant_tweet_vecs.append(tweet_vec)
    print(np.average(all_distances))
    user.features['relevant_tweets'] = relevant_tweets
    user.features['relevant_tweet_vecs'] = relevant_tweet_vecs
    return user


def build_dataset(users):
    global fact_to_words
    fact_topics = build_fact_topics()
    fact_to_words = {r['hash']: [w for w in r['fact_terms'] if w in word_vectors.vocab] for index, r in fact_topics[['hash', 'fact_terms']].iterrows()}
    print(fact_to_words)
    users = Parallel(n_jobs=num_jobs)(
            delayed(get_series_from_user)(user) for i, user in enumerate(users))
    users = sorted(users, key= lambda x: x.user_id)
    return users


def format_training_data(users):
    X = []
    user_order = []
    y = []
    for user in users:
        if 'relevant_tweet_vecs' not in user.features: continue
        for vec in user.features['relevant_tweet_vecs']:
            X.append(vec)
            y.append(user.was_correct)
            user_order.append(user.user_id)
    X = np.asarray(X)
    y = np.asarray(y)
    user_order = np.asarray(user_order)
    print("Average words in tweet: {}".format(sum([len(x) for x in X])/len(X)))
    print(X.shape, y.shape, user_order.shape)
    return X,y,user_order


def keep_n_best_words(X, n = 5000):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    X = vectorizer.fit_transform(X)
    ch2 = SelectKBest(chi2, k=n)
    ch2.fit(X)
    mask = ch2.get_support(indices=True)
    X = [x for x in X for w in x if w in mask]
    return X



def main():
    global bow_corpus
    global word_to_idx
    wn.ensure_loaded()
    bow_corpus = get_corpus()

    bow_corpus_tmp = [w[0] for w in bow_corpus.items() if w[1] > 2 and w[0] in word_vectors.vocab]
    print(len(bow_corpus_tmp))
    word_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_word = {idx: k for k, idx in word_to_idx.items()}

    print("Retrieving data and shaping")
    users = get_users()
    users = [u for u in users if u.tweets]
    users_relevant_tweets = build_dataset(users)
    print("Subselecting best words")
    X,y,user_order = format_training_data(users_relevant_tweets)
    top_words = 50000
    X = keep_n_best_words(X,top_words)

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    max_tweet_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_tweet_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_tweet_length)

    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_tweet_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


if __name__ == "__main__":
    main()


