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

NEW_MODEL = False
NEW_REL_TWEETS = False
DIR = os.path.dirname(__file__) + '../../3_Data/'
word_vectors = KeyedVectors.load_word2vec_format('model_data/word2vec_twitter_model/word2vec_twitter_model.bin',
                                                 binary=True, unicode_errors='ignore')


def train_test_split_on_facts(X, y, user_order, facts_train, users):
    user_to_fact = {user.user_id: user.fact for user in users}
    fact_order = [user_to_fact[uid] for uid in user_order]
    train_indeces = np.asarray([True if f in facts_train else False for idx, f in enumerate(fact_order)])
    X_train = X[train_indeces]
    y_train = y[train_indeces]
    X_test = X[~train_indeces]
    y_test = y[~train_indeces]
    return X_train, X_test, y_train, y_test


def train_test_split_every_user(X, y, user_order, test_size = 0.2):
    user_to_num_instances = defaultdict(lambda: 0)
    user_to_num_instances_train = {u:0 for u in list(set(user_order))}
    for u in user_order: user_to_num_instances[u] += 1
    train_indeces = []
    for u in user_order:
        if user_to_num_instances_train[u] >= (1-test_size)*user_to_num_instances[u]:
            train_indeces.append(False)
        else:
            train_indeces.append(True)
            user_to_num_instances_train[u] += 1
    train_indeces = np.asarray(train_indeces)
    X_train = X[train_indeces]
    y_train = y[train_indeces]
    X_test = X[~train_indeces]
    y_test = y[~train_indeces]
    return X_train, X_test, y_train, y_test


def get_relevant_tweets(user, i=0.8):
    relevant_tweets = []
    if user.features is None: user.features = {}
    if user.fact not in fact_to_words: return user
    if user.tweets is None: return user
    user_fact_words = [fw for fw in fact_to_words[user.fact] if fw in word_vectors.vocab]
    for tweet in user.tweets:
        distance_to_topic = []
        tokens = gt.get_tokenize_text(tweet['text'])
        for token in tokens:
            if token not in word_vectors.vocab: continue
            increment = np.average(word_vectors.distances(token, other_words=user_fact_words))
            distance_to_topic.append(increment)
        if np.average(np.asarray(distance_to_topic)) < i:
            relevant_tweets.append(tweet)
    user.features['relevant_tweets'] = relevant_tweets
    # print(user.user_id, len(user.features['relevant_tweets']))
    return user


def get_relevant_tweets_test_set(user, X_test, i=0.8):
    relevant_tweets = []
    if user.features is None: user.features = {}
    if user.fact not in fact_to_words: return user
    if user.tweets is None: return user
    user_fact_words = [fw for fw in fact_to_words[user.fact] if fw in word_vectors.vocab]
    for tweet in user.tweets:
        distance_to_topic = []
        tokens = gt.get_tokenize_text(tweet['text'])
        for token in tokens:
            if token not in word_vectors.vocab: continue
            increment = np.average(word_vectors.distances(token, other_words=user_fact_words))
            distance_to_topic.append(increment)
        if np.average(np.asarray(distance_to_topic)) < i:
            t_vec = [word_to_idx[t] for t in tokens if t in word_to_idx]
            if t_vec in X_test:
                relevant_tweets.append(tweet)
    user.features['relevant_tweets'] = relevant_tweets
    # print(user.user_id, len(user.features['relevant_tweets']))
    return user


def cred_stance_prediction(this_users):
    def get_support(text, cred):
        sent = sid.polarity_scores(text)['compound']
        if sent > 0.5:
            return cred, True
        if sent < -0.5:
            return 1 - cred, True
        else:
            return float(((sent * cred) + 1) / 2), False

    this_users = this_users.sort_values('fact_text_ts')
    assertions = []
    #assertions.append(float(get_credibility(this_fact['text'].values[0])))

    for idx, u in this_users.iterrows():
        user_cred = u.credibility
        pred, T = get_support(u['fact_text'], user_cred)
        if T:
            assertions.append(pred)
        assertions.append(pred)
    result = [np.average(assertions[:i + 1]) for i in range(len(assertions))]
    return result


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
    cred_to_fact_text = []
    # Maybe this one should be somehow enabled
    #assertions.append(float(get_credibility(this_fact['text'].values[0])))

    for idx, u in this_users.iterrows():
        user_cred = u.credibility
        user_pred = get_support(u, user_cred)
        cred_to_fact_text.append([user_cred,u.fact_text])
        if u.stance != -1:
            assertions.append(user_pred)
        assertions.append(user_pred)
    result = [np.average(assertions[:i + 1]) for i in range(len(assertions))]
    return result, cred_to_fact_text


def feature_cred_stance(this_users):
    def get_support(user, cred):
        if user.stance == 1:
            return cred
        if user.stance == 0:
            return 1 - cred
        else:
            sent = sid.polarity_scores(u['fact_text'])['compound']
            return float(((sent * cred) + 1) / 2)
    # Would only work with a sequential model
    this_users = this_users.sort_values('fact_text_ts')
    creds = []
    stances = []
    # Maybe this one should be somehow enabled
    #assertions.append(float(get_credibility(this_fact['text'].values[0])))

    for idx, u in this_users.iterrows():
        user_cred = u.credibility
        user_pred = get_support(u, user_cred)
        creds.append(user_pred)
        stances = u.stance if u.stance != -1 else (sid.polarity_scores(u['fact_text'])['compound'] +1) / 2
    #result = [round(np.average(assertions[:i + 1])) for i in range(len(assertions))]
    return None


def prebuild_cred(model, user):
    # print(user.user_id)
    def get_credibility(text):
        text = gt.get_tokenize_text(text)
        text = [word_to_idx[w] for w in text if w in word_to_idx]
        text = sequence.pad_sequences([text], maxlen=12)
        probs = model.predict_proba(text)
        return probs
    user_cred = []
    user_cred.append(get_credibility(user.fact_text))
    if user.features is None or 'relevant_tweets' not in user.features: user.credibility = user_cred[0]; return user
    relevant_tweets = user.features['relevant_tweets']
    for idx,tweet in enumerate(relevant_tweets):
        if idx > 200: break
        user_cred.append(get_credibility(tweet['text']))
    user.credibility = np.average(user_cred)
    return user


def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


def store_result(user):
    with open(DIR + 'user_tweets/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')


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

    if NEW_MODEL:
        users = gt.get_users()
        # Prepping lstm model
        top_words = 50000
        X, y, user_order = lstm_cred.get_prebuilt_data()
        X, y, user_order = lstm_cred.balance_classes(X, y, user_order)

        X_train, X_test, y_train, y_test = train_test_split_every_user(X, y, user_order)
        #X_train, X_test, y_train, y_test = train_test_split_on_facts(X, y, user_order, facts_train.values, users)
        #X_train, X_test, y_train, y_test = lstm_cred.train_test_split_on_users(X, y, user_order, users, 100)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        X_train, X_test, word_to_idx = lstm_cred.keep_n_best_words(X_train, y_train, X_test, y_test, idx_to_word, top_words)
        max_tweet_length = 12
        X_train = sequence.pad_sequences(X_train, maxlen=max_tweet_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_tweet_length)

        # Training lstm model
        embedding_vecor_length = 32
        model = Sequential()
        model.add(Embedding(top_words, embedding_vecor_length, input_length=max_tweet_length))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)
        model.save('model_data/cred_model.h5')
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        if NEW_REL_TWEETS:
            print('Building new relevant tweets')
            users = Parallel(n_jobs=num_jobs)(delayed(get_relevant_tweets)(user) for user in users)
            #users = Parallel(n_jobs=num_jobs)(delayed(get_relevant_tweets_test_set)(user, X_test) for user in users)
            user_to_rel_tweet = {user.user_id: user.features['relevant_tweets'] for user in users if 'relevant_tweets' in user.features}
            with open('model_data/relevant_tweets.pkl','wb') as tmpfile:
                pickle.dump(user_to_rel_tweet, tmpfile)
        else:
            with open('model_data/relevant_tweets.pkl','rb') as tmpfile:
                user_to_rel_tweet = pickle.load(tmpfile)
            for user in users: user.features['relevant_tweets'] = user_to_rel_tweet[user.user_id]

        # Build credibility scores for all users on their topic
        print('Computing credibility')
        users = [prebuild_cred(model, u) for u in users]
        users_df = pd.DataFrame([vars(u) for u in users])

        [store_result(u) for u in users]
        with open('model_data/cred_pred_data','wb') as tmpfile:
            pickle.dump({'users':users_df, 'map': word_to_idx}, tmpfile)
    else:
        print('Loading users & model')
        with open('model_data/cred_pred_data','rb') as tmpfile:
            construct = pickle.load(tmpfile)
        users_df = construct['users']
        word_to_idx = construct['map']


    print('Making cred*sent predictions')
    X = []
    y = []
    for idx, hsh in enumerate(facts['hash'].values):
        this_users = users_df[users_df['fact'] == hsh]
        this_x = cred_stance_prediction(this_users)
        this_y = facts['true'].iloc[idx]
        X.append((this_x[-1], np.std(this_x)))
        y.append(int(this_y))
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


    print('Making cred*stance predictions')
    X = []
    y = []
    all_evidence = []
    for idx, hsh in enumerate(facts['hash'].values):
        this_users = users_df[users_df['fact'] == hsh]
        this_x, evidence = only_cred_support_deny_pred(this_users)
        this_y = facts['true'].iloc[idx]
        evidence = sorted(evidence, reverse=True, key=lambda x: x[0])
        #print(facts[facts['hash']==hsh]['text'].values, int(this_y), this_x[-1])
        #print(evidence if len(evidence) <3 else evidence[:3])
        X.append((this_x[-1], np.std(this_x)))
        y.append(int(this_y))
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

    # Pred with faulty stance

    print('Making cred * faulty stance predictions')
    X = []
    y = []
    all_evidence = []
    with open('model_data/faulty_stances.json', 'rb') as tmpfile:
        f_stances = json.load(tmpfile)
    # print(f_stances[:10])
    users_df['true_stance'] = users_df['stance']
    users_df['stance'] = users_df['tweet_id'].map(lambda x: f_stances[x])
    for idx, hsh in enumerate(facts['hash'].values):
        this_users = users_df[users_df['fact'] == hsh]
        this_x, evidence = only_cred_support_deny_pred(this_users)
        this_y = facts['true'].iloc[idx]
        X.append((this_x[-1], np.std(this_x)))
        y.append(int(this_y))
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
