from __future__ import print_function
import os
import sys
from collections import Counter, defaultdict
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from keras.preprocessing import sequence
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
sid = SentimentIntensityAnalyzer()

sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
sys.path.insert(0, os.path.dirname(__file__) + '../1_user_cred_models')
import lstm_pred_credibility as lstm_cred
import getters as gt

DIR = os.path.dirname(__file__) + '../../3_Data/'


def train_test_split_on_facts(X, y, user_order, facts_train):
    user_to_fact = {user.user_id: user.fact for user in users}
    fact_order = [user_to_fact[uid] for uid in user_order]
    train_indeces = np.asarray([True if f in facts_train else False for idx,f in enumerate(fact_order)])
    X_train = X[train_indeces]
    y_train = y[train_indeces]
    X_test = X[~train_indeces]
    y_test = y[~train_indeces]
    return X_train, X_test, y_train, y_test


def cred_fact_prediction(model, hash, facts, transactions, users_df):
    def pred_times_sent(text):
        sent = sid.polarity_scores(text)['compound']
        text = gt.get_tokenize_text(text)
        text = [w for w in text if w in word_to_idx]
        text = sequence.pad_sequences([text], maxlen=12)
        pred = model.predict(text)
        return sent*pred
    assertions = []
    this_fact = facts[facts['hash']==hash]
    this_transactions = transactions[transactions['fact']==hash]
    assertions.append(pred_times_sent(this_fact['text']))
    this_transactions.sort_values('timestamp', inplace=True)
    for idx, tr in this_transactions.iterrows():
        assertions.append(pred_times_sent(tr['text']))
    result = [np.average(assertions[:i] for i in range(len(assertions)))]
    return result


def main():
    global bow_corpus
    global word_to_idx, idx_to_word
    global bow_corpus_top_n
    global users
    wn.ensure_loaded()
    bow_corpus = gt.get_corpus()
    users = gt.get_users()
    users_df = pd.DataFrame([vars(u) for u in users])
    facts = gt.get_fact_topics()
    transactions = gt.get_transactions()

    facts_train, facts_test, _, _ = train_test_split(facts['hash'].values, [0] * len(facts.index))
    facts_train = facts[facts['hash'].isin(facts_train)]
    facts_test = facts[facts['hash'].isin(facts_test)]

    bow_corpus_tmp = [w[0] for w in bow_corpus.items() if w[1] > 2]
    word_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_word = {idx: k for k, idx in word_to_idx.items()}

    # Prepping lstm model
    top_words = 50000
    X, y, user_order = lstm_cred.get_prebuilt_data()
    X, y, user_order = lstm_cred.balance_classes(X, y, user_order)
    X_train, X_test, y_train, y_test = train_test_split_on_facts(X, y, user_order, facts_train.values)
    X_train, X_test, word_to_idx = lstm_cred.keep_n_best_words(X_train, y_train, X_test, y_test, top_words)
    max_tweet_length = 12
    X_train = sequence.pad_sequences(X_train, maxlen=max_tweet_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_tweet_length)
    # Training lstm model
    model = lstm_cred.get_trained_model(X_train, X_test, y_train, y_test)

    pred = []
    y = []
    for idx, fact in facts_test['hash'].values:
        pred.append(cred_fact_prediction(model, fact, facts, transactions, users_df))
        y.append(facts_test['true'].iloc(idx))
        print(pred, y)




if __name__ == "__main__":
    main()
