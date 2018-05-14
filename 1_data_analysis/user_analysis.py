import datetime
import glob
import json
import os
import pickle
import sys
import warnings
from collections import Counter
from string import digits

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
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

import gensim
from gensim.models import KeyedVectors

warnings.filterwarnings("ignore", category=DeprecationWarning)

NEW_CORPUS = False
BUILD_NEW_SPARSE = False

DIR = os.path.dirname(__file__) + '../../3_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))


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
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=decoder)
        yield user


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


def build_sparse_matrix_word2vec(users, word_to_idx):
    def rebuild_sparse():
        print("Building sparse vectors")
        word_vectors = KeyedVectors.load_word2vec_format('model_data/GoogleNews-vectors-negative300.bin', binary=True)
        _, transactions = get_data()
        fact_topics, idx_to_factword = build_fact_topics()

        i = 0
        for user in users:
            if i%50 == 0: print(i)
            user_data = {}
            if not user.tweets: print(user.user_id); continue
            if int(user.was_correct) != -1: y_only_0_1.append(i)
            i += 1
            for t in transactions:
                if user.user_id == t.user_id:
                    user.fact = t.fact
                    transactions.pop(transactions.index(t))
                    break
            user_fact_words = np.array(fact_topics[fact_topics.hash == user.fact]['fact_terms'].as_matrix()[0])
            user_fact_words = [w for w in user_fact_words if w in word_vectors.vocab]
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

            positions.append(this_position)
            data.append(this_data)
            user_order.append(user.user_id)
            y.append(int(user.was_correct))
        with open('model_data/positions_w2v.txt', 'wb') as tmpfile:
            pickle.dump(positions, tmpfile)
        with open('model_data/data_w2v.txt', 'wb') as tmpfile:
            pickle.dump(data, tmpfile)
        with open('model_data/user_w2v.txt', 'wb') as tmpfile:
            pickle.dump(y, tmpfile)
        with open('model_data/order_w2v.txt', 'wb') as tmpfile:
            pickle.dump(user_order, tmpfile)
    y = []
    positions = []
    data = []
    user_order = []
    y_only_0_1 = []

    if not BUILD_NEW_SPARSE:
        print("Using pre-computed sparse")
        with open('model_data/positions_w2v.txt', 'rb') as f:
            positions = pickle.load(f)
        with open('model_data/data_w2v.txt', 'rb') as f:
            data = pickle.load(f)
        with open('model_data/user_w2v.txt', 'rb') as f:
            y = pickle.load(f)
        with open('model_data/order_w2v.txt', 'rb') as f:
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


def build_fact_topics():
    print("Build fact topics")
    def build_feature_vector(terms):
        vec = [0] * len(factword_to_idx)
        for t in terms:
            if t not in factword_to_idx: continue
            vec[factword_to_idx[t]] += 1
        return vec

    fact_file = glob.glob(DIR + 'facts_annotated.json')[0]
    facts_df = pd.read_json(fact_file)
    remove_digits = str.maketrans('', '', digits)
    facts_df['text_parsed'] = facts_df['text'].map(lambda t: tokenize_text(t.translate(remove_digits)))
    facts_df['entities_parsed'] = facts_df['entities'].map(lambda ents:
                                                           [item for sublist in [e['surfaceForm'].lower().split() for e in ents if e['similarityScore'] >= 0.6]
                                                            for item in sublist])
    facts_df['topic'] = facts_df['topic'].map(lambda t: [t])
    facts_df['fact_terms'] = facts_df['text_parsed'] + facts_df['entities_parsed'] + facts_df['topic']
    facts_bow = {}
    for terms in facts_df['fact_terms']:
        for k in terms:
            if k in facts_bow: facts_bow[k] += 1
            else:
                facts_bow[k] = 1
    bow_corpus_tmp = [w[0] for w in facts_bow.items() if w[1] > 2]
    factword_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_factword = {idx: k for k, idx in factword_to_idx.items()}

    facts_df['feature_vector'] = facts_df['fact_terms'].map(lambda terms: build_feature_vector(terms))
    return facts_df, idx_to_factword


def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)


def evaluation(X, y):
    def benchmark(clf):
        clf.fit(X_train_imp, y_train)
        pred = clf.predict(X_test_imp)
        scores = cross_val_score(clf, X_test_imp, y_test, cv=5)
        #neg_log_loss = model_selection.cross_val_score(clf, X, y, cv=5, scoring='roc_auc')

        score = metrics.accuracy_score(y_test, pred)
        # print("accuracy:   %0.3f" % score)
        #print("Logloss: {}, {}").format(neg_log_loss.mean(), neg_log_loss.std())
        print("Cross validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores.mean()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X_train)
    X_train_imp = imp.transform(X_train)
    X_test_imp = imp.transform(X_test)

    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest"),
            (BernoulliNB(alpha=.01), "Bernoulli NB")):
        print('=' * 80)
        print(name)
        results.append([benchmark(clf), clf])
    # Train sparse Naive Bayes classifiers

    for penalty in ["l2", "l1"]:
        print("%s penalty" % penalty.upper())
        for clf, name in (
                (LinearSVC(penalty=penalty, dual=False, tol=1e-3), "Linear SVM"),
                (SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty), "SGDC")):
            print('=' * 80)
            results.append([benchmark(clf), clf])
    return results[np.argmax(np.asarray(results)[:, 0])]


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


def truth_prediction_for_users(word_to_idx, idx_to_word):
    print('Credibility (Was user correct) Prediction using BOWs')
    X_user, y, user_order = build_sparse_matrix_word2vec(get_users(), word_to_idx)

    transformer = TfidfTransformer(smooth_idf=False)
    X_user = transformer.fit_transform(X_user)

    #word_vectors = KeyedVectors.load_word2vec_format('model_data/GoogleNews-vectors-negative300.bin', binary=True)
    ch, pv= chi2(X_user, y)
    # inspect how many words appear in word2vec
    print(sorted([[idx_to_word[idx],p] for idx, p in enumerate(pv)], reverse=True, key=lambda k: k[1])[:200])
    #words_in_vocab = np.asarray(sorted([t[0] for t in top10k if t[0] in word_vectors.vocab]))
    print(X_user.shape)

    ch2 = SelectKBest(chi2, k=10000)
    X_user = ch2.fit_transform(X_user, y)

    svd = TruncatedSVD(20)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = np.asarray(lsa.fit_transform(X_user, y))

    evaluation(X, y)


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
    wn.ensure_loaded()
    if NEW_CORPUS:
        bow_corpus = build_bow_corpus(get_users())
        save_corpus(bow_corpus)
    else:
        bow_corpus = get_corpus()

    bow_corpus_tmp = [w[0] for w in bow_corpus.items() if w[1] > 2]
    word_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_word = {idx: k for k, idx in word_to_idx.items()}

    # corpus_analysis(bow_corpus, word_to_idx, idx_to_word)
    # temporal_analysis(get_users())
    # cluster_users_on_tweets(get_users(), word_to_idx, idx_to_word)
    truth_prediction_for_users(word_to_idx, idx_to_word)
    # lda_analysis(get_users())


if __name__ == "__main__":
    main()
