from __future__ import print_function

import datetime
import os
import sys
from dateutil import parser
from collections import Counter, defaultdict
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from sklearn import metrics
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import Imputer, normalize, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sid = SentimentIntensityAnalyzer()

sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
sys.path.insert(0, os.path.dirname(__file__) + '../1_user_cred_models')
import getters as gt

DIR = os.path.dirname(__file__) + '../../5_Data/'


def get_features(fact, transactions, users):
    if fact['true'] == 'unknown': print(fact); return
    this_transactions = transactions[transactions['fact'] == fact['hash']]
    if this_transactions.shape[0] == 0: print(fact.hash); return

    avg_friends = []
    avg_followers = []
    avg_status_cnt = []
    avg_reg_age = []
    avg_links = []
    avg_sent_pos = []
    avg_sent_neg = []
    avg_emoticons = []
    avg_questionM = []
    avg_mentions = []
    avg_personal_pronoun_first = []
    fr_has_url = []
    avg_sentiment = []
    share_most_freq_author = []
    lvl_size = ''
    matched_users = 0

    emoji_pattern = re.compile(
        "(:\(|:\))|"
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)

    link_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    pronouns = ['I', 'you', 'he', 'she', 'it', 'they', 'we', 'me', 'him', 'her', 'its', 'our', 'us', 'them', 'my',
                'your', 'his', 'hers', 'yours', 'theirs', 'mine', 'ours']

    for idx, tr in this_transactions.iterrows():
        tokenized_text = gt.get_tokenize_text(tr['text'])
        if len(tokenized_text) == 0: tokenized_text = [0]
        chars = [c for c in tr['text']]
        avg_questionM.append(1 if '?' in tr['text'] else 0)
        avg_personal_pronoun_first.append(1 if tokenized_text[0] in pronouns else 0)
        sent_score = sid.polarity_scores(tr['text'])['compound']
        avg_sentiment.append(sent_score)
        avg_sent_pos.append(1 if sent_score > 0.5 else 0)
        avg_sent_neg.append(1 if sent_score < 0.5 else 0)
        avg_emoticons.append(len(re.findall(emoji_pattern, tr['text'])))
        avg_links.append(len(re.findall(link_pattern, tr['text'])))
        fr_has_url.append(1 if len(re.findall(link_pattern, tr['text'])) != 0 else 0)
        avg_mentions.append(len([c for c in chars if c == '@']))
        lvl_size = idx

        # if tr['user_id'] in users['user_id'].values: print(tr['user_id'])
        user = [u for u in users if int(u.user_id) == int(tr['user_id'])]
        if len(user) <1: print(tr['user_id']); continue
        user = user[0]
        if user.features == None: print(user.user_id); continue

        matched_users += 1
        avg_friends.append(int(user.features['friends']) if 'friends' in user.features else 0)
        avg_followers.append(int(user.features['followers']) if 'followers' in user.features else 0)
        avg_status_cnt.append(int(user.features['statuses_count']) if 'statuses_count' in user.features else 0)
        avg_reg_age.append(int((datetime.datetime.now().replace(tzinfo=None) - parser.parse(
            user.features['created_at']).replace(tzinfo=None)).days if 'created_at' in user.features else 0))

    if matched_users == 0: matched_users = 1
    num_transactions = this_transactions.shape[0]

    avg_emoticons = 1.0 * sum(avg_emoticons) / (num_transactions)
    avg_mentions = sum(avg_mentions) / (num_transactions)
    avg_links = 1.0 * sum(avg_links) / (num_transactions)
    avg_questionM = 1.0 * sum(avg_questionM) / (num_transactions)
    avg_sent_pos = 1.0 * sum(avg_sent_pos) / (num_transactions)
    avg_sent_neg = 1.0 * sum(avg_sent_neg) / (num_transactions)
    avg_sentiment = 1.0 * sum(avg_sentiment) / (num_transactions)
    avg_personal_pronoun_first = sum(avg_personal_pronoun_first) * 1.0 / (num_transactions)
    fr_has_url = 1.0 * sum(fr_has_url) / (num_transactions)
    share_most_freq_author = 1 / num_transactions
    lvl_size = lvl_size

    avg_friends = 1.0 * sum(avg_friends) / matched_users
    avg_reg_age = 1.0 * sum(avg_reg_age) / matched_users
    avg_followers = 1.0 * sum(avg_followers) / matched_users
    avg_status_cnt = 1.0 * sum(avg_status_cnt) / matched_users

    return {
        'hash': fact['hash'],
        'topic': fact['topic'],
        'source_tweet': fact['source_tweet'],
        'text': fact['text'],
        'y': int(fact['true']),
        'avg_mentions': avg_mentions,
        'avg_emoticons': avg_emoticons,
        'avg_links': avg_links,
        'avg_questionM': avg_questionM,
        'avg_personal_pronoun_first': avg_personal_pronoun_first,
        'avg_sent_pos': avg_sent_pos,
        'avg_sent_neg': avg_sent_neg,
        'avg_sentiment': avg_sentiment,
        'fr_has_url': fr_has_url,
        'share_most_freq_author': share_most_freq_author,
        'lvl_size': lvl_size,
        'avg_followers': avg_followers,
        'avg_friends': avg_friends,
        'avg_status_cnt': avg_status_cnt,
        'avg_reg_age': avg_reg_age
    }


def evaluation(X_train, X_test, y_train, y_test):
    def benchmark(clf):
        scores = cross_val_score(clf, np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)), cv=5)

        clf.fit(X_train_imp, y_train)
        pred = clf.predict(X_test_imp)

        score = metrics.accuracy_score(y_test, pred)
        precision, recall, fscore, sup = precision_recall_fscore_support(y_test, pred, average='macro')
        print("Rumor prediction: Accuracy: %0.3f, Precision: %0.3f, Recall: %0.3f, F1 score: %0.3f" % (
            score, precision, recall, fscore))
        print("\t Cross validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return fscore, scores.mean()

    print('&' * 80)
    # print("Evaluation")

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X_train)
    X_train_imp = imp.transform(X_train)
    X_test_imp = imp.transform(X_test)

    clf = DecisionTreeClassifier(random_state=0)
    results = benchmark(clf)
    return results


def main():
    global users
    wn.ensure_loaded()
    users = gt.get_users(DIR)
    facts = gt.get_fact_topics(DIR)
    transactions = gt.get_transactions(DIR)
    print(transactions.describe())

    tr_hsh = transactions['fact'].values
    cond = facts['hash'].isin(tr_hsh )
    facts = facts[cond]
    facts = pd.DataFrame([get_features(fact, transactions, users) for idx, fact in facts.iterrows() if fact['true'] != 'unknown'])
    print(facts.describe())

    features = ['avg_mentions', 'avg_emoticons', 'avg_links', 'avg_questionM', 'avg_personal_pronoun_first',
                'avg_sent_pos', 'avg_sent_neg', 'avg_sentiment', 'fr_has_url', 'share_most_freq_author', 'lvl_size',
                'avg_followers', 'avg_friends', 'avg_status_cnt', 'avg_reg_age']

    X = facts[list(features)].values
    y = facts['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(Counter(y_train), Counter(y_test), X_train.shape)

    evaluation(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()