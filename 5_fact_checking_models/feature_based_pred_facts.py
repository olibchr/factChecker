from __future__ import print_function

import datetime
import multiprocessing
import os
import pickle
import sys
from dateutil import parser
from collections import Counter, defaultdict
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from nltk.corpus import wordnet as wn
from sklearn import metrics
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, normalize, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
sys.path.insert(0, os.path.dirname(__file__) + '../1_user_cred_models')
import getters as gt

DIR = os.path.dirname(__file__) + '../../3_Data/'
NEW_DATA = False
if NEW_DATA:
    sid = SentimentIntensityAnalyzer()
    num_cores = multiprocessing.cpu_count()
    num_jobs = round(num_cores * 3 / 4)


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

    avg_len = []
    avg_words = []
    all_characters = []
    avg_unique_char = []
    retweets = []
    tweets_that_are_retweet = []
    avg_special_symbol = []
    tweets_that_are_reply = []
    avg_hashtags = []
    avg_exlamationM = []
    avg_multiQueExlM = []
    avg_upperCase = []
    avg_count_distinct_hashtags = []
    most_common_weekday = []
    most_common_hour = []
    avg_count_distinct_words = []
    avg_verified = []
    avg_time_retweet = []
    avg_len_description = []
    avg_len_name = []

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
        avg_links.extend(re.findall(link_pattern, tr['text']))
        fr_has_url.append(1 if len(re.findall(link_pattern, tr['text'])) != 0 else 0)
        avg_mentions.append(len([c for c in chars if c == '@']))
        lvl_size = idx
        avg_len.append(len(tr['text']))
        avg_words.append(len(tokenized_text))
        avg_count_distinct_words.extend(tokenized_text)
        chars = [c for c in tr['text']]
        all_characters.extend(chars)
        avg_unique_char.append(len(set(chars)))
        avg_hashtags.append(len([c for c in chars if c == '#']))
        avg_mentions.append(len([c for c in chars if c == '@']))
        avg_special_symbol.append(len(re.findall('[^0-9a-zA-Z *]', tr['text'])))
        avg_questionM.append(1 if '?' in tr['text'] else 0)
        avg_exlamationM.append(1 if '!' in tr['text'] else 0)
        avg_multiQueExlM.append(
            1 if len(re.findall('/[?]/', tr['text'])) + len(re.findall('/[!]/', tr['text'])) > 1 else 0)
        avg_upperCase.append(len(re.findall('/[A-Z]/', tr['text'])))
        avg_count_distinct_hashtags.append((len(re.findall('/[#]/', tr['text']))))
        most_common_weekday.append(tr['timestamp'].day)
        most_common_hour.append(tr['timestamp'].hour)

        # if tr['user_id'] in users['user_id'].values: print(tr['user_id'])
        user = [u for u in users if int(u.user_id) == int(tr['user_id'])]
        if len(user) < 1: print(tr['user_id']); continue
        user = user[0]
        if user.features == None: print(user.user_id); continue

        matched_users += 1
        avg_friends.append(int(user.features['friends']) if 'friends' in user.features else 0)
        avg_followers.append(int(user.features['followers']) if 'followers' in user.features else 0)
        avg_status_cnt.append(int(user.features['statuses_count']) if 'statuses_count' in user.features else 0)
        avg_reg_age.append(int((datetime.datetime.now().replace(tzinfo=None) - parser.parse(
            user.features['created_at']).replace(tzinfo=None)).days if 'created_at' in user.features else 0))

        avg_verified.append(1 if 'verified' in user.features and user.features['verified'] == 'true' else 0)
        avg_time_retweet.append(int(user.avg_time_to_retweet))
        avg_len_description.append(len(user.features['description']) if 'description' in user.features else 0)
        avg_len_name.append(len(user.features['name']) if 'name' in user.features else 0)

    if matched_users == 0: matched_users = 1
    num_transactions = this_transactions.shape[0]

    avg_emoticons = 1.0 * sum(avg_emoticons) / (num_transactions)
    avg_mentions = sum(avg_mentions) / (num_transactions)
    avg_links = 1.0 * len(set(avg_links)) / (num_transactions)
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

    avg_len = 1.0 * sum(avg_len) / num_transactions
    avg_words = 1.0 * sum(avg_words) / num_transactions
    avg_unique_char = 1.0 * sum(avg_unique_char) / num_transactions
    avg_hashtags = sum(avg_hashtags) / num_transactions
    avg_retweets = 1.0 * sum(retweets) / num_transactions
    avg_special_symbol = sum(avg_special_symbol) / num_transactions
    avg_exlamationM = 1.0 * sum(avg_exlamationM) / num_transactions
    avg_multiQueExlM = 1.0 * sum(avg_multiQueExlM) / num_transactions
    avg_upperCase = 1.0 * sum(avg_upperCase) / num_transactions
    avg_count_distinct_hashtags = 1.0 * sum(avg_count_distinct_hashtags) / num_transactions
    most_common_weekday = 1.0 * sum(most_common_weekday) / num_transactions
    most_common_hour = 1.0 * sum(most_common_hour) / num_transactions
    avg_count_distinct_words = 1.0 * len(set(avg_count_distinct_words)) / num_transactions
    avg_verified = 1.0 * sum(avg_verified) / num_transactions
    avg_time_retweet = 1.0 * sum(avg_time_retweet) / num_transactions
    avg_len_description = 1.0 * sum(avg_len_description) / num_transactions
    avg_len_name = 1.0 * sum(avg_len_name) / num_transactions

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
        'avg_reg_age': avg_reg_age,

        'avg_len': avg_len,
        'avg_words': avg_words,
        'avg_unique_char': avg_unique_char,
        'avg_hashtags': avg_hashtags,
        'avg_retweets': avg_retweets,
        'avg_special_symbol': avg_special_symbol,
        'avg_exlamationM': avg_exlamationM,
        'avg_multiQueExlM': avg_multiQueExlM,
        'avg_upperCase': avg_upperCase,
        'avg_count_distinct_hashtags': avg_count_distinct_hashtags,
        'most_common_weekday': most_common_weekday,
        'most_common_hour': most_common_hour,
        'avg_count_distinct_words': avg_count_distinct_words,
        'avg_verified': avg_verified,
        'avg_time_retweet': avg_time_retweet,
        'avg_len_description': avg_len_description,
        'avg_len_name': avg_len_name
    }


def feature_pred(features, chik, ldak):
    global users
    wn.ensure_loaded()
    facts = gt.get_fact_topics(DIR)

    if NEW_DATA:
        users = gt.get_users(DIR)
        transactions = gt.get_transactions(DIR)
        print(transactions.describe())

        tr_hsh = transactions['fact'].values
        cond = facts['hash'].isin(tr_hsh)
        cond2 = facts['true'] != 'unknown'
        facts = facts[cond & cond2]
        facts = Parallel(n_jobs=num_jobs)(
            delayed(get_features)
            (fact, transactions[transactions['fact'] == fact['hash']], [u for u in users if int(u.user_id) in
                                                                        list(transactions[
                                                                                 transactions['fact'] == fact['hash']][
                                                                                 'user_id'].values)])
            for idx, fact in facts.iterrows())
        facts = pd.DataFrame(facts)
        with open('model_data/feature_data', 'wb') as tmpfile:
            pickle.dump(facts, tmpfile)
    else:
        with open('model_data/feature_data', 'rb') as tmpfile:
            facts = pickle.load(tmpfile)

    X = facts[list(features)].values
    y = facts['y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=ldak), SVC(C=1, gamma=1))
    std_clf.fit(X_train, y_train)
    pred_test_std = std_clf.predict(X_test)
    precision, recall, fscore, sup = precision_recall_fscore_support(y_test, pred_test_std, average='macro')
    score = metrics.accuracy_score(y_test, pred_test_std)
    print("Accuracy: %0.3f, Precision: %0.3f, Recall: %0.3f, F1 score: %0.3f" % (
        score, precision, recall, fscore))
    scores = cross_val_score(std_clf, X, y, cv=3)
    print("\t Cross validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()


def main():
    import copy
    not_in_list = ['share_most_freq_author', 'avg_reg_age', 'avg_questionM', 'avg_emoticons', 'avg_friends',
                   'avg_words', 'avg_personal_pronoun_first', 'avg_followers',
                   'avg_len_description', 'avg_hashtags', 'avg_status_cnt', 'avg_mentions', 'avg_exlamationM',
                   'avg_verified', 'avg_multiQueExlM', 'avg_upperCase', 'avg_count_distinct_hashtags']
    features = ['avg_links', 'avg_sent_neg', 'avg_sentiment', 'fr_has_url', 'lvl_size',
                'avg_len', 'avg_unique_char', 'avg_special_symbol',
                'avg_len_name', 'avg_time_retweet', 'avg_count_distinct_words', 'avg_sent_pos'
                ]
    r = []
    for i in range(10):
        r.append(feature_pred(features, 15, 10))
    print(np.argmax(r))


if __name__ == "__main__":
    main()
