from __future__ import print_function

import multiprocessing
import os
import pickle
import sys
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

from sklearn.model_selection import train_test_split
from keras.models import load_model
from gensim.models import KeyedVectors

sid = SentimentIntensityAnalyzer()

sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
sys.path.insert(0, os.path.dirname(__file__) + '../1_user_cred_models')
import lstm_pred_credibility as lstm_cred
import getters as gt

num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

NEW_MODEL = True
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


def get_relevant_tweets(user, i=0.8):
    relevant_tweets = []
    if user.fact not in fact_to_words: return user
    if user.tweets is None: return user
    if user.features is None: user.features = {}
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
    print(user.user_id, len(user.features['relevant_tweets']))
    return user


def cred_fact_prediction(model, this_fact, this_users):
    def get_credibility(text):
        text = gt.get_tokenize_text(text)
        text = [word_to_idx[w] for w in text if w in word_to_idx]
        text = sequence.pad_sequences([text], maxlen=12)
        return model.predict_proba(text)

    def get_support(text, cred):
        sent = sid.polarity_scores(text)['compound']
        return float(((sent * cred) + 1) / 2)

    this_users.sort_values('fact_text_ts', inplace=True)

    assertions = []
    assertions.append(float(get_credibility(this_fact['text'].values[0])))

    for idx, u in this_users.iterrows():
        user_cred = []
        user_cred.append(get_credibility(u['fact_text']))
        relevant_tweets = u.features['relevant_tweets']
        for idx,tweet in enumerate(relevant_tweets):
            if idx > 200: break
            user_cred.append(get_credibility(tweet['text']))

        user_cred = np.average(user_cred)
        assertions.append(get_support(u['fact_text'], user_cred))
    result = [round(np.average(assertions[:i + 1])) for i in range(len(assertions))]
    return result, hash


def main():
    global bow_corpus
    global word_to_idx, idx_to_word, fact_to_words
    global bow_corpus_top_n
    wn.ensure_loaded()
    print('Grabbing Data')
    bow_corpus = gt.get_corpus()
    facts = gt.get_fact_topics()
    transactions = gt.get_transactions()
    facts = facts[facts['true'] != 'unknown']

    facts_train, facts_test, _, _ = train_test_split(facts['hash'].values, [0] * len(facts.index))
    facts_train = facts[facts['hash'].isin(facts_train)]
    facts_test = facts[facts['hash'].isin(facts_test)]

    bow_corpus_tmp = [w[0] for w in bow_corpus.items() if w[1] > 2]
    word_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_word = {idx: k for k, idx in word_to_idx.items()}
    fact_to_words = {r['hash']: [w for w in r['fact_terms']] for index, r in facts[['hash', 'fact_terms']].iterrows()}
    print(len(fact_to_words))


    if NEW_MODEL:
        print('Building new users & model')
        users = gt.get_users()
        users = Parallel(n_jobs=num_jobs)(delayed(get_relevant_tweets)(user) for user in users)
        users_df = pd.DataFrame([vars(u) for u in users])
        with open('model_data/cred_pred_user_data','wb') as tmpfile:
            pickle.dump(users_df, tmpfile)

        # Prepping lstm model
        top_words = 50000
        X, y, user_order = lstm_cred.get_prebuilt_data()
        X, y, user_order = lstm_cred.balance_classes(X, y, user_order)
        X_train, X_test, y_train, y_test = train_test_split_on_facts(X, y, user_order, facts_train.values, users)
        X_train, X_test, y_train, y_test = lstm_cred.train_test_split_on_users(X, y, user_order, users, 100)
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
    else:
        print('Loading users & model')
        model = load_model('model_data/cred_model.h5')
        with open('model_data/cred_pred_user_data','rb') as tmpfile:
            users_df = pickle.load(tmpfile)

    print('Making predictions')
    pred = []
    y = []
    for idx, hsh in enumerate(facts_test['hash'].values):
        this_fact = facts[facts['hash'] == hsh]
        this_users = users_df[users_df['fact'] == hsh]
        pred_n = cred_fact_prediction(model, this_fact, this_users)

        this_y = facts_test['true'].iloc[idx]
        pred.append(pred_n[-1])
        y.append(this_y)

    score = metrics.accuracy_score(y, pred)
    precision, recall, fscore, sup = metrics.precision_recall_fscore_support(y, pred, average='macro')
    print("Rumors: Accuracy: %0.3f, Precision: %0.3f, Recall: %0.3f, F1 score: %0.3f" % (
        score, precision, recall, fscore))


if __name__ == "__main__":
    main()
