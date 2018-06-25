import datetime
import glob
import json
import multiprocessing
import os
import pickle
import sys, re
import warnings
from collections import Counter, defaultdict
from itertools import cycle
from string import digits

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from numpy.core.multiarray import interp
from scipy.sparse import lil_matrix, csr_matrix
from sklearn import metrics, preprocessing
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, normalize, LabelBinarizer
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassificationReport
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import coo_matrix, vstack
import seaborn as sns
import random
import scikitplot as skplt

sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
from decoder import decoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

NEW_CORPUS = True
BUILD_NEW_SPARSE = True

DIR = os.path.dirname(__file__) + '../../../5_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))

num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

# global Vars
user_order, users = [],[]
word_to_idx = {}
fact_to_words = {}
bow_corpus_cnt = {}
if BUILD_NEW_SPARSE or NEW_CORPUS:
    word_vectors = KeyedVectors.load_word2vec_format('model_data/GoogleNews-vectors-negative300.bin', binary=True)
word_vectors = 0#KeyedVectors.load_word2vec_format('model_data/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')


def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def tokenize_text(text, only_retweets=False):
    tokenizer = RegexpTokenizer(r'\w+')
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


def save_corpus(bow_corpus):
    with open('model_data/bow_corpus.json', 'w') as out_file:
        out_file.write(json.dumps(bow_corpus, default=datetime_converter) + '\n')


def build_bow_corpus(users):
    print("Building a new Bow corpus")
    bow_corpus_cnt = {}
    for user in users:
        if user.tweets is None:
            print(user.user_id)
            continue
        for tweet in user.tweets:
            # Tweets <text, created_at, *quoted_status
            tokens = tokenize_text(tweet['text'])
            for token in tokens:
                if token in bow_corpus_cnt:
                    bow_corpus_cnt[token] += 1
                else:
                    bow_corpus_cnt[token] = 1
    return bow_corpus_cnt


def build_user_vector(user, i):
    global fact_to_words
    if i % 50 == 0: print(i)
    user_data = {}
    if not user.tweets: print("PROBLEM DUE TO: {}".format(user.user_id)); return
    user_fact_words = fact_to_words[user.fact]
    if len(user_fact_words) == 0:
        print("%%%%%%%%")
        print(user.user_id)
        print(user.fact)
        print(user_fact_words)
        return

    user_fact_words = fact_to_words[user.fact]
    # If X doesnt need to be rebuild, comment out
    for tweet in user.tweets:
        tokens = tokenize_text(tweet['text'], only_retweets=False)
        for token in tokens:
            if token not in word_to_idx: continue
            if token not in word_vectors.vocab: continue
            if len(user_fact_words) == 0: user_fact_words = [token]
            increment = 1 - np.average(word_vectors.distances(token, other_words=user_fact_words))
            if increment > 1: increment = 1
            if increment < 0: increment = 0

            if token in user_data:
                user_data[word_to_idx[token]] += increment
            else:
                user_data[word_to_idx[token]] = increment

    this_position = []
    this_data = []
    for tuple in user_data.items():
        this_position.append(tuple[0])
        this_data.append(tuple[1])

    package = {
        'index': i,
        'positions': this_position,
        'data': this_data,
        'user_id': user.user_id,
        'y': int(user.was_correct)
    }
    return package


def build_sparse_matrix_word2vec(users, retweets_only=False):
    def rebuild_sparse(users):
        global fact_to_words
        print("Building sparse vectors")
        _, transactions = get_data()
        fact_topics = build_fact_topics()
        fact_to_words = {r['hash']: [w for w in r['fact_terms'] if w in word_vectors.vocab] for index, r in fact_topics[['hash', 'fact_terms']].iterrows()}
        users = sorted(users, key=lambda x: x.fact_text_ts)
        for user in users:
            if not user.tweets: users.pop(users.index(user))
            for t in transactions:
                if user.user_id == t.user_id:
                    user.fact = t.fact
                    transactions.pop(transactions.index(t))
                    break
            if user.fact is None: print(user.user_id)
        if retweets_only:
            classification_data = Parallel(n_jobs=num_jobs)(
            delayed(build_user_vector)(user, i) for i, user in enumerate(users))
            return sorted([x for x in classification_data if x != None], key=lambda x: x['index'])

        classification_data = Parallel(n_jobs=num_jobs)(
            delayed(build_user_vector)(user, i) for i, user in enumerate(users))
        classification_data = [x for x in classification_data if x != None]
        classification_data = sorted(classification_data, key=lambda x: x['index'])
        with open('model_data/classification_data_w2v', 'wb') as tmpfile:
            pickle.dump(classification_data, tmpfile)
        return classification_data

    positions = []
    data = []
    y = []
    user_order = []
    classification_data = []

    if not BUILD_NEW_SPARSE and not retweets_only:
        with open('model_data/classification_data_w2v', 'rb') as f:
            classification_data = pickle.load(f)
    else:
        classification_data = rebuild_sparse(users)

    for item in classification_data:
        positions.append(item['positions'])
        data.append(item['data'])
        user_order.append(item['user_id'])
        y.append(item['y'])

    # Only considering supports and denials [0,1], not comments etc. [-1]
    mask = [el != -1 for el in y]
    positions = np.asarray(positions)[mask]
    data = np.asarray(data)[mask]
    y = np.asarray(y)[mask]
    user_order = np.asarray(user_order)[mask]

    X = lil_matrix((len(data), len(word_to_idx)))
    X.rows = positions
    X.data = data
    X.tocsr()
    print(X.shape, y.shape, user_order.shape)
    return X, y, user_order


def build_fact_topics():
    print("Build fact topics")
    fact_file = glob.glob(DIR + 'facts.json')[0]
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


def build_alternative_features(users, user_order):
    X = []
    for u_id in user_order:
        user = [u for u in users if u.user_id == u_id][0]
        followers = int(user.features['followers']) if 'followers' in user.features else 0
        friends = int(user.features['friends']) if 'friends' in user.features else 0
        verified = 1 if 'verified' in user.features and user.features['verified'] == 'true' else 0
        status_cnt = int(user.features['statuses_count']) if 'statuses_count' in user.features else 0
        pos_words = int(user.features['pos_words']) if 'pos_words' in user.features else 0
        neg_words = int(user.features['neg_words']) if 'neg_words' in user.features else 0
        sent_avg = int(user.sent_tweets_avg)
        time_retweet = int(user.avg_time_to_retweet)
        X.append([followers, friends, verified, status_cnt, pos_words, neg_words, sent_avg, time_retweet])
    print(X[:5])
    return X


def train_test_split_on_facts(X, y, user_order, users, n):
    fact_file = glob.glob(DIR + 'facts_annotated.json')[0]
    facts_df = pd.read_json(fact_file)
    facts_hsh = list(facts_df['hash'].as_matrix())
    f_train, f_test, _, _ = train_test_split(facts_hsh, [0] * len(facts_hsh), test_size=min(0.15 + n / 30, 0.5))

    user_to_fact = {user.user_id: user.fact for user in users}
    user_order_fact = [user_to_fact[u_id] for u_id in user_order]

    # build a mask
    f_train_mask = []
    fact_to_n = defaultdict(lambda: 0)
    for user_fact in user_order_fact:
        # always true if in train set
        if user_fact in f_train:
            f_train_mask.append(True);
            continue
        # true to add n samples of rumor to train set
        elif fact_to_n[user_fact] < n:
            f_train_mask.append(True)
            fact_to_n[user_fact] += 1
            continue
        # otherwise false if in test set
        else:
            f_train_mask.append(False)
    f_train_mask = np.asarray(f_train_mask)

    # f_train_mask = np.asarray([True if f in f_train else False for f in user_order_hashed_fact])
    X_train = X[f_train_mask == True]
    X_test = X[f_train_mask == False]
    y_train = y[f_train_mask == True]
    y_test = y[f_train_mask == False]
    print("Shapes after splitting")

    for user in users:
        if user.user_id not in user_order: continue
        i = np.where(user_order == user.user_id)[0][0]
        assert str(user.fact) == str(user_order_fact[i])
        assert int(user.user_id) == int(user_order[i])
        assert int(user.was_correct) == int(y[i])

    print("Training Set: {}, {}".format(X_train.shape, y_train.shape))
    print("Testing Set: {}, {}".format(X_test.shape, y_test.shape))
    return X_train, X_test, np.asarray(y_train), np.asarray(y_test)


def evaluation(X, y, X_train=None, X_test=None, y_train=None, y_test=None):
    def benchmark(clf):
        clf.fit(X_train_imp, y_train)
        pred = clf.predict(X_test_imp)
        # print(X_test_imp.shape, y_test.shape, pred.shape)

        score = metrics.accuracy_score(y_test, pred)
        precision, recall, fscore, sup = precision_recall_fscore_support(y_test, pred, average='macro')
        print("Unknown rumors: Accuracy: %0.3f, Precision: %0.3f, Recall: %0.3f, F1 score: %0.3f" % (
            score, precision, recall, fscore))

        clf.fit(X_train_imp2, y_train2)
        pred2 = clf.predict(X_test_imp2)
        score2 = metrics.accuracy_score(y_test2, pred2)
        precision2, recall2, fscore2, sup2 = precision_recall_fscore_support(y_test2, pred2, average='macro')
        print("Random split: Accuracy: %0.3f, Precision: %0.3f, Recall: %0.3f, F1 score: %0.3f" % (
            score2, precision2, recall2, fscore2))

        acc_scores = cross_val_score(clf, X, y, cv=3)
        pr_scores = cross_val_score(clf, X, y, scoring='precision', cv=3)
        re_scores = cross_val_score(clf, X, y, scoring='recall', cv=3)
        f1_scores = cross_val_score(clf, X, y, scoring='f1', cv=3)
        print("\t Cross validated Accuracy: %0.3f (+/- %0.3f)" % (acc_scores.mean(), acc_scores.std() * 2))
        print("\t Cross validated Precision: %0.3f (+/- %0.3f)" % (pr_scores.mean(), pr_scores.std() * 2))
        print("\t Cross validated Recall: %0.3f (+/- %0.3f)" % (re_scores.mean(), re_scores.std() * 2))
        print("\t Cross validated F1: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

        # skplt.metrics.plot_roc_curve(y_train, pred)
        # plt.show()

        # classification_analysis(X,y, clf.predict(X))
        return [fscore, fscore2, acc_scores.mean()]

    print('&' * 80)
    print("Evaluation")

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X_train)
    X_train_imp = imp.transform(X_train)
    X_test_imp = imp.transform(X_test)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X_train2)
    X_train_imp2 = imp.transform(X_train2)
    X_test_imp2 = imp.transform(X_test2)
    results = []

    model = LinearSVC(penalty='l2', dual=False, tol=1e-3)
    results.append(benchmark(model))

    return results


def visualizations(X,y):

    svd = TruncatedSVD(2)
    lsa = make_pipeline(svd)
    X_2d = lsa.fit_transform(X,y)
    X_2d = normalize(X_2d, axis=0)

    # 2d plot of X
    X2d_df = pd.DataFrame({'x1': X_2d[:, 0], 'x2': X_2d[:, 1], 'y': y})
    sns.lmplot(data=X2d_df, x='x1', y='x2', hue='y')
    plt.show()


def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def balance_classes(X,y, user_order):
    bigger_class = 0 if (Counter(y)[0]-Counter(y)[1]) > 0 else 1
    diff = abs(Counter(y)[0]-Counter(y)[1])

    k_add = random.sample(list(np.where(y==1-bigger_class)[0]), int(diff))
    X = vstack([X, X[k_add]])
    y = np.append(y, y[k_add])
    user_order = np.append(user_order, user_order[k_add])

    # k_del = random.sample(list(np.where(y==bigger_class)[0]), int(diff/2))
    # X = delete_rows_csr(X, k_del)
    # y = np.delete(y,k_del,0)
    # user_order = np.delete(user_order,k_del,0)
    return X,y, user_order


def truth_prediction_for_users(users, idx_to_word, chik, svdk, N):
    global y, user_order
    print('%' * 100)
    print('Credibility (Was user correct) Prediction using BOWs')
    print(chik, svdk, N)

    X, y, user_order = build_sparse_matrix_word2vec(users)
    X,y,user_order = balance_classes(X,y,user_order)

    transformer = TfidfTransformer(smooth_idf=True)
    std_scale = preprocessing.StandardScaler(with_mean=False)
    svd = TruncatedSVD(svdk)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    # X = SelectKBest(chi2, k=chik).fit_transform(X, y)
    # X = transformer.fit_transform(X, y)
    # X = np.asarray(lsa.fit_transform(X, y))

    # X_train, X_test, y_train, y_test = train_test_split_on_facts(X, y, user_order, users, n=N)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #X_train = ch2.fit_transform(X_train,y_train)
    #X_train = std_scale.fit_transform(X_train, y_train)
    #X_train = transformer.fit_transform(X_train, y_train)
    #X_train = np.asarray(lsa.fit_transform(X_train, y_train))

    #X_test = ch2.transform(X_test)
    #X_test = std_scale.transform(X_test)
    #X_test = transformer.transform(X_test)
    #X_test = np.asarray(lsa.transform(X_test))

    svd = TruncatedSVD(2)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd)
    X_2d = lsa.fit_transform(X, y)
    X_2d = normalize(X_2d, axis=0)

    # 2d plot of X
    # X2d_df = pd.DataFrame({'x1': X_2d[:, 0], 'x2': X_2d[:, 1], 'y': y})
    # sns.lmplot(data=X2d_df, x='x1', y='x2', hue='y')
    # plt.show()

    std_clf = make_pipeline(SelectKBest(chi2, k=chik), TfidfTransformer(smooth_idf=True), preprocessing.StandardScaler(with_mean=False), TruncatedSVD(svdk), SVC(C=1, gamma=1))
    std_clf.fit(X_train, y_train)
    pred_test_std = std_clf.predict(X_test)
    precision, recall, fscore, sup = precision_recall_fscore_support(y_test, pred_test_std, average='macro')
    score = metrics.accuracy_score(y_test, pred_test_std)
    print("Random split: Accuracy: %0.3f, Precision: %0.3f, Recall: %0.3f, F1 score: %0.3f" % (
        score, precision, recall, fscore))
    acc_scores = cross_val_score(std_clf, X, y, cv=3)
    pr_scores = cross_val_score(std_clf, X, y, scoring='precision', cv=3)
    re_scores = cross_val_score(std_clf, X, y, scoring='recall', cv=3)
    f1_scores = cross_val_score(std_clf, X, y, scoring='f1', cv=3)
    print("\t Cross validated Accuracy: %0.3f (+/- %0.3f)" % (acc_scores.mean(), acc_scores.std() * 2))
    print("\t Cross validated Precision: %0.3f (+/- %0.3f)" % (pr_scores.mean(), pr_scores.std() * 2))
    print("\t Cross validated Recall: %0.3f (+/- %0.3f)" % (re_scores.mean(), re_scores.std() * 2))
    print("\t Cross validated F1: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

    print(Counter(y), Counter(y_train), Counter(y_test))
    return evaluation(X, y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def main():
    global bow_corpus
    global word_to_idx
    global users
    wn.ensure_loaded()
    if NEW_CORPUS:
        bow_corpus = build_bow_corpus(get_users())
        save_corpus(bow_corpus)
    else:
        bow_corpus = get_corpus()

    bow_corpus_tmp = [w[0] for w in bow_corpus.items() if w[1] > 2]
    word_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_word = {idx: k for k, idx in word_to_idx.items()}

    users = get_users()

    results = []
    N = 0
    # for chik, svdk in exp:
    #    r= []
    for N in range(15):
        results.append(truth_prediction_for_users(users, idx_to_word, 10000, 20, N))
    print(np.average(np.asarray(results), axis=1))


if __name__ == "__main__":
    main()
