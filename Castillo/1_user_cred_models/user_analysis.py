import datetime
import glob
import json
import multiprocessing
import os
import pickle
import sys, re
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
import seaborn as sns
sns.set(style="ticks")


sys.path.insert(0, os.path.dirname(__file__) + '../../2_helpers')
from decoder import decoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

NEW_CORPUS = False
BUILD_NEW_SPARSE = False

DIR = os.path.dirname(__file__) + '../../5_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))

num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

# global Vars
word_to_idx = {}
word_vectors = None
fact_to_words = {}
bow_corpus_cnt = {}


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
        if int(user.was_correct) != -1:
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
    def preprocessing(X,y):
        print(X.shape, y.shape)
        y_r = y == 1
        y_r = y_r.nonzero()[0]
        X_r = X[y_r, :]
        X_r_s = X_r.sum(axis=1)
        X_r_s = np.argsort(X_r_s)[-10:]
        print('Common terms for users that are correct: {}'.format([idx_to_word[xrs] for xrs in X_r_s]))
        X_r_r = X_r.sum() * 1.0 / len(y_r)
        # print('# of terms for correct users on avg: {}'.format(X_r_r))

        y_w = y == 0
        y_w = y_w.nonzero()[0]
        X_w = X[y_w, :]
        X_w_s = X_w.sum(axis=1)
        X_w_s = np.argsort(X_w_s)[-10:]
        print('Common terms for users that are incorrect: {}'.format([idx_to_word[xws] for xws in X_w_s]))
        Y_w_r = X_w.sum() * 1.0 / len(y_w)
        # print('# of terms for incorrect users on avg: {}'.format(Y_w_r))

        transformer = TfidfTransformer(smooth_idf=False)
        X_weighted = transformer.fit_transform(X)

        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X_pca = lsa.fit_transform(X_weighted)

        sns.pairplot(pd.DataFrame({'X1': X_pca[0], 'X2': X_pca[1], 'y':y}), hue='y')

    print("KMeans Clustering")
    X, y, user_order = build_sparse_matrix(users, word_to_idx)
    users_df = pd.DataFrame([vars(s) for s in users])
    X = X.toarray()


    users_df['X'] = users_df['user_id'].map(lambda uid: X[np.where(user_order==int(uid))[0]])
    users_df['y'] = users_df['user_id'].map(lambda uid: y[np.where(user_order==int(uid))[0]])
    print(users_df['y'])
    for f in set(users_df['fact'].values):
        this_f_users = users_df[users_df['fact'] == f]
        this_X = preprocessing(this_f_users['X'].values, this_f_users['y'].values)
        print("Training KMEans")
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(this_X)

        for cl in set(kmeans.labels_):
            this_cl = kmeans.labels_ == cl
            X_cl = X.toarray()[this_cl, :]
            X_cl = X_cl.sum(axis=1)
            X_cl = np.argsort(X_cl)[:10]

            print("# in this cluster: {}".format(len([cl for cl in this_cl if cl == True])))
            print("% of correct users in class: {}".format(sum(y[this_cl]) * 1.0 / len(y[this_cl])))
            print('Common terms for users in this cluster: {}'.format([idx_to_word[xcl] for xcl in X_cl]))


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

    corpus_analysis(bow_corpus, word_to_idx, idx_to_word)
    # temporal_analysis(get_users())

    cluster_users_on_tweets(users, word_to_idx, idx_to_word)
    # lda_analysis(get_users())


if __name__ == "__main__":
    main()
