import datetime
import json, glob, time, random
import sys, os
import numpy as np
from dateutil import parser
import pickle

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import re, nltk
from nltk.corpus import wordnet as wn
#from User import User
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

NEW_CORPUS = True

DIR = os.path.dirname(__file__) + '../../3_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))


def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def user_decoder(obj):
    if 'user_id' not in obj.keys(): return obj
    # <user_id, tweets, fact, transactions, credibility, controversy>
    return User(obj['user_id'], obj['tweets'], obj['fact'], obj['transactions'], obj['credibility'],
                obj['controversy'])


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
    BUILD_NEW_SPARSE = True
    if not BUILD_NEW_SPARSE:
        with open('positions.txt', 'rb') as f:
            positions = pickle.load(f)
        with open('data.txt', 'rb') as f:
            data = pickle.load(f)
    else:
        for user in users:
            y.append(user.user_id)
            user_data = {}
            if not user.tweets: continue
            for tweet in user.tweets:
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
    print(len(data), len(word_to_idx))
    X = lil_matrix((len(data), len(word_to_idx)))
    X.rows = positions
    X.data = data
    X.tocsr()
    return X


def k_means_clustering(users, word_to_idx, idx_to_word):
    print("KMeans Clustering")
    X = build_sparse_matrix(users, word_to_idx)
    print("Training K means")
    kmeans = KMeans(n_clusters=100)
    kmeans.fit(X)
    print(kmeans.labels_[:100])
    print("Training DBSCAN")
    dbscan = DBSCAN(eps=0.05, min_samples=50)
    dbscan.fit(X.toarray())
    print(dbscan.labels_[:100])


def temporal_analysis(users):
    print("Temporal Analysis")
    t_deltas = []
    for user in users:
        if not user.tweets: t_deltas.append(0); continue
        datetimes = [parser.parse(tweet['created_at']) for tweet in user.tweets]
        timedeltas = [datetimes[i - 1] - datetimes[i] for i in range(1, len(datetimes))]
        if len(timedeltas) == 0: continue
        average_timedelta = sum(timedeltas, datetime.timedelta(0)) / len(timedeltas)
        t_deltas.append(average_timedelta)
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
    else:
        bow_corpus = get_corpus()
    save_corpus(bow_corpus)

    bow_corpus_tmp = [w[0] for w in bow_corpus.items() if w[1] > 50]
    word_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_word = {idx: k for k, idx in word_to_idx.items()}

    corpus_analysis(bow_corpus, word_to_idx, idx_to_word)
    # temporal_analysis(get_users())
    k_means_clustering(get_users(), word_to_idx, idx_to_word)


if __name__ == "__main__":
    main()
