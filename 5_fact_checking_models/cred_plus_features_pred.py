from __future__ import print_function

import multiprocessing
import json
import os
import pickle
import sys
from datetime import datetime
from collections import Counter, defaultdict
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nltk.corpus import wordnet as wn
from sklearn import metrics
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import load_model
from gensim.models import KeyedVectors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import types
import tempfile
import keras.models

sid = SentimentIntensityAnalyzer()

sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
sys.path.insert(0, os.path.dirname(__file__) + '../1_user_cred_models')
import lstm_pred_credibility as lstm_cred
import getters as gt

num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 7 / 8)

DIR = os.path.dirname(__file__) + '../../3_Data/'


def only_cred_support_deny_pred(this_users):
    def get_support(user, cred):
        if user.stance == 1:
            return cred
        if user.stance == 0:
            return 1 - cred
        else:
            sent = sid.polarity_scores(u['fact_text'])['compound']
            return float(((sent * cred) + 1) / 2)

    this_users = this_users.sort_values('fact_text_ts')
    assertions = []
    # Maybe this one should be somehow enabled
    # assertions.append(float(get_credibility(this_fact['text'].values[0])))

    for idx, u in this_users.iterrows():
        user_cred = u.credibility
        user_pred = get_support(u, user_cred)
        if u.stance != -1:
            assertions.append(user_pred)
        assertions.append(user_pred)
    result = [round(np.average(assertions[:i + 1])) for i in range(len(assertions))]
    return result


def main():
    global bow_corpus
    global word_to_idx, idx_to_word, fact_to_words
    global bow_corpus_top_n
    wn.ensure_loaded()
    print('Grabbing Data')
    bow_corpus = gt.get_corpus()
    facts = gt.get_fact_topics()
    facts = facts[facts['true'] != 'unknown']

    bow_corpus_tmp = [w[0] for w in bow_corpus.items() if w[1] > 2]
    word_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_word = {idx: k for k, idx in word_to_idx.items()}
    fact_to_words = {r['hash']: [w for w in r['fact_terms']] for index, r in facts[['hash', 'fact_terms']].iterrows()}

    # Credibility data
    print('Loading users & model')
    with open('model_data/cred_pred_data', 'rb') as tmpfile:
        construct = pickle.load(tmpfile)
    users_df = construct['users']
    word_to_idx = construct['map']
    # Feature data
    with open('model_data/feature_data', 'rb') as tmpfile:
        fact_features = pickle.load(tmpfile)
    features = ['avg_links', 'avg_sent_neg', 'avg_sentiment', 'fr_has_url', 'lvl_size', 'avg_len', 'avg_special_symbol',
                'avg_time_retweet', 'avg_count_distinct_words', 'avg_sent_pos', 'cred_pred', 'cred_pred_std']

    print('Making cred*stance +best features predictions')
    facts['cred_pred'] = facts['hash'].map(lambda x: only_cred_support_deny_pred(users_df[users_df['fact'] == x]))
    facts['cred_pred_std'] = facts['cred_pred'].map(lambda x: np.std(x))
    facts['cred_pred'] = facts['cred_pred'].map(lambda x: x[-1])
    facts = facts.set_index('hash').join(fact_features.set_index('hash'), rsuffix='_other')
    X = facts[features].values
    y = facts['y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    std_clf = make_pipeline(StandardScaler(), SVC(C=1, gamma=1))
    std_clf.fit(X_train, y_train)
    pred = std_clf.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)
    precision, recall, fscore, sup = metrics.precision_recall_fscore_support(y_test, pred, average='macro')
    print("Rumors: Accuracy: %0.3f, Precision: %0.3f, Recall: %0.3f, F1 score: %0.3f" % (
        score, precision, recall, fscore))
    acc_scores = cross_val_score(std_clf, X, y, cv=3)
    pr_scores = cross_val_score(std_clf, X, y, scoring='precision', cv=3)
    re_scores = cross_val_score(std_clf, X, y, scoring='recall', cv=3)
    f1_scores = cross_val_score(std_clf, X, y, scoring='f1', cv=3)
    print("\t Cross validated Accuracy: %0.3f (+/- %0.3f)" % (acc_scores.mean(), acc_scores.std() * 2))
    print("\t Cross validated Precision: %0.3f (+/- %0.3f)" % (pr_scores.mean(), pr_scores.std() * 2))
    print("\t Cross validated Recall: %0.3f (+/- %0.3f)" % (re_scores.mean(), re_scores.std() * 2))
    print("\t Cross validated F1: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))


if __name__ == "__main__":
    main()
