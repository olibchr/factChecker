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
from scipy.sparse import lil_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
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
word_vectors = None
fact_to_words = {}
if BUILD_NEW_SPARSE:
    word_vectors = KeyedVectors.load_word2vec_format('model_data/GoogleNews-vectors-negative300.bin', binary=True)


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


def get_user_documents(users):
    X = []
    y = []
    for user in users:
        if not user.tweets: continue
        X.append(' '.join([tweet['text'] for tweet in user.tweets]))
        y.append(int(user.was_correct))
    return np.array(X), np.array(y)


def save_corpus(bow_corpus):
    with open('model_data/bow_corpus.json', 'w') as out_file:
        out_file.write(json.dumps(bow_corpus, default=datetime_converter) + '\n')


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


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


def build_sparse_matrix(users, word_to_idx):
    def rebuild_sparse():
        print("Building sparse vectors")
        i = 0
        for user in users:
            user_data = {}
            if not user.tweets: print(user.user_id); continue
            if int(user.was_correct) != -1: y_only_0_1.append(i)
            i += 1
            # If X doesnt need to be rebuild, comment out
            for tweet in user.tweets:
                tokens = tokenize_text(tweet['text'], only_retweets=False)
                for token in tokens:
                    if token not in word_to_idx: continue
                    if token in user_data:
                        user_data[word_to_idx[token]] += 1
                    else:
                        user_data[word_to_idx[token]] = 1
            this_position = []
            this_data = []
            for tuple in user_data.items():
                this_position.append(tuple[0])
                this_data.append(tuple[1])

            positions.append(this_position)
            data.append(this_data)
            user_order.append(user.user_id)
            y.append(int(user.was_correct))
        with open('model_data/positions.txt', 'wb') as tmpfile:
            pickle.dump(positions, tmpfile)
        with open('model_data/data.txt', 'wb') as tmpfile:
            pickle.dump(data, tmpfile)
        with open('model_data/user.txt', 'wb') as tmpfile:
            pickle.dump(y, tmpfile)
        with open('model_data/order.txt', 'wb') as tmpfile:
            pickle.dump(user_order, tmpfile)

    y = []
    positions = []
    data = []
    user_order = []
    y_only_0_1 = []
    if not BUILD_NEW_SPARSE:
        with open('model_data/positions.txt', 'rb') as f:
            positions = pickle.load(f)
        with open('model_data/data.txt', 'rb') as f:
            data = pickle.load(f)
        with open('model_data/user.txt', 'rb') as f:
            y = pickle.load(f)
        with open('model_data/order.txt', 'rb') as f:
            user_order = pickle.load(f)
        y_only_0_1 = [idx for idx, u in enumerate([u for u in users if u.tweets]) if int(u.was_correct) != -1]
    else:
        rebuild_sparse()
    print(len(data), len(word_to_idx), len(y))
    # Only considering supports and denials [0,1], not comments etc. [-1]
    positions = np.asarray(positions)[y_only_0_1]
    data = np.asarray(data)[y_only_0_1]
    y = np.asarray(y)[y_only_0_1]
    user_order = np.asarray(user_order)[y_only_0_1]

    X = lil_matrix((len(data), len(word_to_idx)))
    X.rows = positions
    X.data = data
    X.tocsr()
    return X, np.array(y), np.array(user_order)


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

    # Extend topic description for text that is similar to topic
    # for token in tokenize_text(user.fact_text):
    #     if token not in word_vectors.vocab: continue
    #     user_to_fact_dist = np.average(word_vectors.distances(token, other_words=user_fact_words))
    #     # todo: test value
    #     if user_to_fact_dist < 0.5:
    #         fact_to_words[user.fact] += [token]

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


def build_user_vector_retweets_topic_independent(user, i):
    global fact_to_words
    if i % 50 == 0: print(i)
    user_data = {}
    if not user.tweets: print("PROBLEM DUE TO: {}".format(user.user_id)); return

    # If X doesnt need to be rebuild, comment out
    for tweet in user.tweets:
        tokens = tokenize_text(tweet['text'], only_retweets=True)
        for token in tokens:
            if token not in word_to_idx: continue
            if token in user_data:
                user_data[word_to_idx[token]] += 1
            else:
                user_data[word_to_idx[token]] = 1

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
            delayed(build_user_vector_retweets_topic_independent)(user, i) for i, user in enumerate(users))
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

    if not BUILD_NEW_SPARSE:
        with open('model_data/classification_data_w2v', 'rb') as f:
            classification_data = pickle.load(f)
            # with open('model_data/positions_w2v', 'rb') as f:
            #     positions = pickle.load(f)
            # with open('model_data/data_w2v', 'rb') as f:
            #     data = pickle.load(f)
            # with open('model_data/user_w2v', 'rb') as f:
            #     y = pickle.load(f)
            # with open('model_data/order_w2v', 'rb') as f:
            #     user_order = pickle.load(f)
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
    # print(Counter(list(X[:,0])))
    # print(Counter(list(X[:,1])))
    # print(Counter(list(X[:,2])))
    # print(Counter(list(X[:,3])))
    # print(Counter(list(X[:,4])))
    # print(Counter(list(X[:,5])))
    # print(Counter(list(X[:,6])))
    print(X[:5])
    return X


def train_test_split_on_facts(X, y, user_order, users, n):
    fact_file = glob.glob(DIR + 'facts_annotated.json')[0]
    facts_df = pd.read_json(fact_file)
    facts_hsh = list(facts_df['hash'].as_matrix())
    f_train, f_test, _, _ = train_test_split(facts_hsh, [0] * len(facts_hsh), test_size=max(0.2 + n / 30, 0.8))

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


def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)


def evaluation(X, y, X_train=None, X_test=None, y_train=None, y_test=None):
    def benchmark(clf):
        scores = cross_val_score(clf, X, y, cv=5)

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

        print("\t Cross validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        fpr, tpr, thresholds = roc_curve(y_test2, pred2)
        roc_auc = auc(fpr, tpr)
        # plt.title('Receiver Operating Characteristic')
        #
        # plt.plot(fpr, tpr, 'b',
        # label='AUC = %0.2f'% roc_auc)
        # plt.legend(loc='lower right')
        # plt.plot([0,1],[0,1],'r--')
        # plt.xlim([-0.1,1.2])
        # plt.ylim([-0.1,1.2])
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.show()
        match = [1 if t == p else 0 for t, p in zip(y_test2, pred2)]

        return fscore, fscore2, scores.mean()

    print('&' * 80)
    print("Evaluation")

    if X_train is None:
        print("No pre-split data given")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

    # for clf, name in (
    #         (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
    #         (Perceptron(n_iter=50), "Perceptron"),
    #         (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
    #         (KNeighborsClassifier(n_neighbors=10), "kNN"),
    #         (RandomForestClassifier(n_estimators=100), "Random forest"),
    #         (BernoulliNB(alpha=.01), "Bernoulli NB")):
    #     print('=' * 80)
    #     print(name)
    #     results.append([benchmark(clf)])

    # Train sparse SVM likes
    for penalty in ["l2", "l1"]:
        print("%s penalty" % penalty.upper())
        for clf, name in (
                (LinearSVC(penalty=penalty, dual=False, tol=1e-3), "Linear SVM"),
                (SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty), "SGDC")):
            print('=' * 80)
            print(name)
            results.append([benchmark(clf)])
    return results


def cluster_users_on_tweets(users, word_to_idx, idx_to_word):
    print("KMeans Clustering")
    X, y, _ = build_sparse_matrix(users, word_to_idx)
    print(X.shape, y.shape)
    y_r = y == 1
    y_r = y_r.nonzero()[0]
    X_r = X.toarray()[y_r, :]
    X_r_s = X_r.sum(axis=1)
    X_r_s = np.argsort(X_r_s)[:10]
    print('Common terms for users that are correct: {}'.format([idx_to_word[xrs] for xrs in X_r_s]))
    X_r_r = X_r.sum() * 1.0 / len(y_r)
    # print('# of terms for correct users on avg: {}'.format(X_r_r))

    y_w = y == 0
    y_w = y_w.nonzero()[0]
    X_w = X.toarray()[y_w, :]
    X_w_s = X_w.sum(axis=1)
    X_w_s = np.argsort(X_w_s)[:10]
    print('Common terms for users that are incorrect: {}'.format([idx_to_word[xws] for xws in X_w_s]))
    Y_w_r = X_w.sum() * 1.0 / len(y_w)
    # print('# of terms for incorrect users on avg: {}'.format(Y_w_r))

    transformer = TfidfTransformer(smooth_idf=False)
    X_weighted = transformer.fit_transform(X)

    svd = TruncatedSVD(20)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_pca = lsa.fit_transform(X_weighted)

    print("Training KMEans")
    kmeans = KMeans(n_clusters=20)
    kmeans.fit(X_pca)

    for cl in set(kmeans.labels_):
        this_cl = kmeans.labels_ == cl
        X_cl = X.toarray()[this_cl, :]
        X_cl = X_cl.sum(axis=1)
        X_cl = np.argsort(X_cl)[:10]

        print("# in this cluster: {}".format(len([cl for cl in this_cl if cl == True])))
        print("% of correct users in class: {}".format(sum(y[this_cl]) * 1.0 / len(y[this_cl])))
        print('Common terms for users in this cluster: {}'.format([idx_to_word[xcl] for xcl in X_cl]))


def truth_prediction_for_users(users, chik, svdk, N):
    print('%' * 100)
    print('Credibility (Was user correct) Prediction using BOWs')
    print(chik, svdk, N)

    X, y, user_order = build_sparse_matrix_word2vec(users)

    transformer = TfidfTransformer(smooth_idf=True)
    ch2 = SelectKBest(chi2, k=chik)
    svd = TruncatedSVD(svdk)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = transformer.fit_transform(X)
    X = ch2.fit_transform(X, y)
    X = np.asarray(lsa.fit_transform(X, y))

    X_alt = build_alternative_features(users, user_order)
    X_alt = transformer.fit_transform(X)
    ch2, pval = chi2(X_alt, y)
    print(pval)
    X_alt = build_sparse_matrix_word2vec(users, retweets_only=True)
    X_alt = transformer.fit_transform(X)
    ch2, pval = chi2(X_alt, y)
    print(pval)

    X_train, X_test, y_train, y_test = train_test_split_on_facts(X, y, user_order, users, n=N)
    print(Counter(y), Counter(y_train), Counter(y_test))
    return evaluation(X, y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # X_train = transformer.fit_transform(X_train)
    # X_test = transformer.transform(X_test)

    # X_train = ch2.fit_transform(X_train, y_train)
    # X_test = ch2.transform(X_test)

    # X_train = np.asarray(lsa.fit_transform(X_train, y_train))
    # X_test = np.asarray(lsa.transform(X_test))


def lda_analysis(users):
    n_samples = 2000
    n_features = 1000
    n_components = 50
    n_top_words = 20
    print("Constructing user docs")
    X, y = get_user_documents(users)
    print("TF fitting user docs")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(X)

    print("Training LDA model")
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    user_topics = []
    for user_vector in tf:
        user_topics.append(lda.transform(user_vector))
    flat_user_topics = [item for user_topic in user_topics for item in user_topic]
    # https://stackoverflow.com/questions/20984841/topic-distribution-how-do-we-see-which-document-belong-to-which-topic-after-doi
    threshold = sum(flat_user_topics) / len(flat_user_topics)
    user_topics_thresholded = []
    for ut in user_topics:
        user_topics_thresholded.append([idx for idx, score in enumerate(ut) if score > threshold])


def temporal_analysis(users):
    print("Temporal Analysis")
    t_deltas_r = []
    t_deltas_w = []
    t_deltas = []
    i = 0
    for user in users:
        i += 1
        if i % 100 == 0: print("progress: {}".format(i))
        if i > 500: break
        if not user.tweets:
            # t_deltas.append(0)
            continue
        datetimes = [parser.parse(tweet['created_at']) for tweet in user.tweets]
        timedeltas = [datetimes[i - 1] - datetimes[i] for i in range(1, len(datetimes))]
        if len(timedeltas) == 0: continue
        average_timedelta = round((sum(timedeltas, datetime.timedelta(0)) / len(timedeltas)).seconds / 60)
        t_deltas.append(average_timedelta)
        if user.was_correct == 1:
            t_deltas_r.append(average_timedelta)
        else:
            t_deltas_w.append(average_timedelta)
    n, bins, patches = plt.hist(t_deltas, 50, normed=1, facecolor='green', alpha=0.75)
    plt.show()


def corpus_analysis(bow_corpus, word_to_idx, idx_to_word):
    print("BOW Analysis")
    print("Top occuring terms: {}".format(sorted(bow_corpus.items(), reverse=True, key=lambda w: w[1])[:20]))
    print("Amoount of words in the corpus: {}".format(len(bow_corpus.items())))
    print("Amount of words in the corpus after filtering: {}".format(len(word_to_idx.items())))
    print("Amount of tweets ")


def main():
    global bow_corpus
    global word_to_idx
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

    # corpus_analysis(bow_corpus, word_to_idx, idx_to_word)
    # temporal_analysis(get_users())
    # cluster_users_on_tweets(get_users(), word_to_idx, idx_to_word)
    exp = [(1000, 5), (5000, 5), (10000, 5), (20000, 5), (50000, 5),
           (1000, 10), (5000, 10), (10000, 10), (20000, 10), (50000, 10),
           (1000, 20), (5000, 20), (10000, 20), (20000, 20), (50000, 20),
           (1000, 50), (5000, 50), (10000, 50), (20000, 50), (50000, 50),
           (1000, 100), (5000, 100), (10000, 100), (20000, 100), (50000, 100)]
    results = []
    N = 0
    # for chik, svdk in exp:
    #    r= []
    # for N in range(15):
    results.append(truth_prediction_for_users(users, 10000, 20, N))
    #    results.append(np.average(np.asarray(r), axis=1))
    print(np.average(np.asarray(results), axis=1))
    # lda_analysis(get_users())


if __name__ == "__main__":
    main()
