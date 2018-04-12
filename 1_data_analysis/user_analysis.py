import datetime
import json, glob, time, random
import sys, os
import numpy as np
from dateutil import parser
import pickle
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.decomposition import NMF, LatentDirichletAllocation

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import re, nltk
from nltk.corpus import wordnet as wn
from User import User
from Transaction import Transaction
from Fact import Fact
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

NEW_CORPUS = False
BUILD_NEW_SPARSE = False

DIR = os.path.dirname(__file__) + '../../3_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))


def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def fact_decoder(obj):
    # <RUMOR_TPYE, HASH, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND, SOURCE_TWEET>
    return Fact(obj['rumor_type'], obj['topic'], obj['text'], obj['true'], obj['proven_false'],
                obj['is_turnaround'], obj['source_tweet'], hash=obj['hash'])


def transaction_decoder(obj):
    # <sourceId, id, user_id, fact, timestamp, stance, weight>
    return Transaction(obj['sourceId'], obj['id'], obj['user_id'], obj['fact'], obj['timestamp'], obj['stance'],
                       obj['weight'])


def user_decoder(obj):
    if 'user_id' not in obj.keys(): return obj
    # <user_id, tweets, fact, transactions, credibility, controversy>
    return User(obj['user_id'], obj['tweets'], obj['fact'], obj['transactions'], obj['credibility'],
                obj['controversy'], obj['features'], obj['was_correct'])


def get_data():
    fact_file = glob.glob(DIR + 'facts.json')[0]
    transactions_file = glob.glob(DIR + 'factTransaction.json')[0]
    facts = json.load(open(fact_file), object_hook=fact_decoder)
    transactions = json.load(open(transactions_file), object_hook=transaction_decoder)
    return facts, transactions


def get_users():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    print('{} users'.format(len(user_files)))
    if len(user_files) < 10: print('WRONG DIR?')
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=user_decoder)
        yield user


def get_corpus():
    corpus_file = glob.glob('bow_corpus.json')[0]
    bow_corpus = json.loads(open(corpus_file).readline())
    return bow_corpus


def build_bow_corpus(users):
    print("Building a new Bow corpus")
    bow_corpus_cnt = {}
    tokenizer = RegexpTokenizer(r'\w+')
    for user in users:
        if user.tweets is None:
            print(user.user_id)
            continue
        for tweet in user.tweets:
            # Tweets <text, created_at, *quoted_status
            tokens = [WNL.lemmatize(i.lower()) for i in tokenizer.tokenize(tweet['text']) if
                      i.lower() not in NLTK_STOPWORDS]
            for token in tokens:
                if token in bow_corpus_cnt:
                    bow_corpus_cnt[token] += 1
                else:
                    bow_corpus_cnt[token] = 1
    return bow_corpus_cnt


def build_sparse_matrix(users, word_to_idx):
    tokenizer = RegexpTokenizer(r'\w+')
    y = []
    positions = []
    data = []
    if not BUILD_NEW_SPARSE:
        with open('positions.txt', 'rb') as f:
            positions = pickle.load(f)
        with open('data.txt', 'rb') as f:
            data = pickle.load(f)
        with open('user.txt', 'rb') as f:
            y = pickle.load(f)
    else:
        print("Building sparse vectors")
        for user in users:
            user_data = {}
            if not user.tweets: continue
            for tweet in user.tweets:
                y.append(int(user.was_correct))
                tokens = [WNL.lemmatize(i.lower()) for i in tokenizer.tokenize(tweet['text']) if
                          i.lower() not in NLTK_STOPWORDS]
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
        with open('positions.txt', 'wb') as tmpfile:
            pickle.dump(positions, tmpfile)
        with open('data.txt', 'wb') as tmpfile:
            pickle.dump(data, tmpfile)
        with open('user.txt', 'wb') as tmpfile:
            pickle.dump(y, tmpfile)
    print(len(data), len(word_to_idx))
    X = lil_matrix((len(data), len(word_to_idx)))
    X.rows = positions
    X.data = data
    X.tocsr()
    return X, np.array(y)


def get_user_documents(users):
    X = []
    y = []
    for user in users:
        if not user.tweets: continue
        X.append(' '.join([tweet['text'] for tweet in user.tweets]))
        y.append(int(user.was_correct))
    return np.array(X), np.array(y)


def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)


def cluster_users_on_tweets(users, word_to_idx, idx_to_word):
    print("KMeans Clustering")
    X, y = build_sparse_matrix(users, word_to_idx)
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


def truth_prediction_for_users(users, word_to_idx):
    X, y = build_sparse_matrix(users, word_to_idx)

    transformer = TfidfTransformer(smooth_idf=False)
    X_weighted = transformer.fit_transform(X)

    svd = TruncatedSVD(20)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_pca = lsa.fit_transform(X_weighted)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

    results = []
    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                           tol=1e-3), X_train, y_train, X_test, y_test))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty), X_train, y_train, X_test, y_test))


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


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


def save_corpus(bow_corpus):
    with open('bow_corpus.json', 'w') as out_file:
        out_file.write(json.dumps(bow_corpus, default=datetime_converter) + '\n')


def main():
    global bow_corpus
    wn.ensure_loaded()
    if NEW_CORPUS:
        bow_corpus = build_bow_corpus(get_users())
        save_corpus(bow_corpus)
    else:
        bow_corpus = get_corpus()

    bow_corpus_tmp = [w[0] for w in bow_corpus.items() if w[1] > 50]
    word_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_word = {idx: k for k, idx in word_to_idx.items()}

    # corpus_analysis(bow_corpus, word_to_idx, idx_to_word)
    # temporal_analysis(get_users())
    # cluster_users_on_tweets(get_users(), word_to_idx, idx_to_word)
    # truth_prediction_for_users(get_users(), word_to_idx)
    lda_analysis(get_users())


if __name__ == "__main__":
    main()
