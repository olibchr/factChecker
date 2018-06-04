from __future__ import print_function
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
import re

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
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
from decoder import decoder

warnings.filterwarnings("ignore", category=DeprecationWarning)
# fix random seed for reproducibility
np.random.seed(7)

BUILD_NEW_DATA = False

DIR = os.path.dirname(__file__) + '../../3_Data/'

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))

num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

# global Vars
word_to_idx, idx_to_word = {}, {}
fact_to_words = {}
bow_corpus_top_n = []
#if BUILD_NEW_SPARSE:
word_vectors = KeyedVectors.load_word2vec_format('model_data/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')


def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def tokenize_text(text, only_retweets=False):
    tokenizer = RegexpTokenizer(r'\w+')
    links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for l in links: text = text.replace(l,' evidence ')
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
    print('Kept {} users'.format(len(users)))
    return users


def get_corpus():
    corpus_file = glob.glob('model_data/bow_corpus.json')[0]
    bow_corpus = json.loads(open(corpus_file).readline())
    return bow_corpus


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


def lda_analysis(users):
    n_features = 1000
    n_components = 50
    n_top_words = 20
    print("Constructing user docs")
    X = [[tweet['text'] for tweet in user.tweets] for user in users]
    X = [tweet for sublist in X for tweet in sublist]
    print(X[:5])
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
    for topic_idx, topic in enumerate(lda.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([tf_feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def get_series_from_user(user):
    relevant_tweets= []
    relevant_tweet_vecs = []
    all_distances = []
    user_fact_words = fact_to_words[user.fact]
    for tweet in user.tweets:
        tokens = tokenize_text(tweet['text'], only_retweets=False)
        # print(tokens, user_fact_words)
        distance_to_topic = []
        for token in tokens:
            if token not in word_to_idx: continue
            if token not in word_vectors.vocab: continue
            increment = np.average(word_vectors.distances(token, other_words=user_fact_words))
            distance_to_topic.append(increment)
        distance_to_topic = np.asarray(distance_to_topic)
        distance_to_topic = np.average(distance_to_topic)
        all_distances.append(float(distance_to_topic))
        if distance_to_topic < 0.8:
            relevant_tweets.append(tweet)
            tweet_vec = [word_to_idx[t] for t in tokens if t in word_to_idx]
            relevant_tweet_vecs.append(tweet_vec)
    print(len(relevant_tweets))
    user.features['relevant_tweets'] = relevant_tweets
    user.features['relevant_tweet_vecs'] = relevant_tweet_vecs
    return user


def build_dataset(users):
    global fact_to_words
    fact_topics = build_fact_topics()
    fact_to_words = {r['hash']: [w for w in r['fact_terms'] if w in word_vectors.vocab] for index, r in fact_topics[['hash', 'fact_terms']].iterrows()}
    #print(fact_to_words)
    users = Parallel(n_jobs=num_jobs)(
            delayed(get_series_from_user)(user) for i, user in enumerate(users))
    users = sorted(users, key= lambda x: x.user_id)
    return users


def format_training_data(users):
    X = []
    user_order = []
    y = []
    for user in users:
        if 'relevant_tweet_vecs' not in user.features:
            print('&§$%&§$%&§$%&')
            print(user.user_id)
            continue
        for vec in user.features['relevant_tweet_vecs']:
            X.append(vec)
            y.append(user.was_correct)
            user_order.append(user.user_id)
    X = np.asarray(X)
    y = np.asarray(y)
    user_order = np.asarray(user_order)
    print("Average words in tweet: {}".format(sum([len(x) for x in X])/len(X)))
    print("Shapes:")
    print(X.shape, y.shape, user_order.shape)
    construct = {
        'X': X,
        'y': y,
        'user_order': user_order
    }
    with open('model_data/lstm_data','wb') as tmpfile:
        pickle.dump(construct, tmpfile)
    return X, y, user_order


def get_prebuilt_data():
    with open('model_data/lstm_data','rb') as tmpfile:
        construct = pickle.load(tmpfile)
    X = construct['X']
    y = construct['y']
    user_order = construct['user_order']
    return X,y,user_order


def keep_n_best_words(X, y, n = 5000):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)

    X_words = [' '.join([idx_to_word[w] for w in x]) for x in X]
    X_words = vectorizer.fit_transform(X_words)
    ch2 = SelectKBest(chi2, k=n)
    ch2.fit(X_words, y)
    mask = ch2.get_support(indices=True)
    vocab = vectorizer.vocabulary_
    vocab = {k:v for k,v in vocab.items() if v in mask}
    vocab_new_indexed = {k[0]:idx for idx, k in enumerate(vocab.items())}
    print("Vocabulary length: {}".format(len(vocab_new_indexed)))

    # only keep words that are selected by chi2
    X = [[vocab_new_indexed[idx_to_word[w]] for w in x if idx_to_word[w] in vocab_new_indexed] for x in X]
    print("Average words in tweet: {}".format(sum([len(x) for x in X])/len(X)))
    return X, vocab_new_indexed


def train_test_split_on_users(X, y, user_order, users, n):
    u_train, u_test, _, _ = train_test_split(list(set(user_order)), [0] * len(set(user_order)))

    # build a mask
    u_train_mask = []
    user_to_n = defaultdict(lambda: 0)
    for uid in user_order:
        # always true if in train set
        if uid in u_train:
            u_train_mask.append(True);
            continue
        # true to add n samples of rumor to train set
        elif user_to_n[uid] < n:
            u_train_mask.append(True)
            user_to_n[uid] += 1
            continue
        # otherwise false if in test set
        else:
            u_train_mask.append(False)
    u_train_mask = np.asarray(u_train_mask)

    print(u_train_mask.shape, np.asarray(X).shape, np.asarray(y).shape, u_train_mask[:5])
    X_train = X[u_train_mask == True]
    X_test = X[u_train_mask == False]
    y_train = y[u_train_mask == True]
    y_test = y[u_train_mask == False]
    print("Shapes after splitting")

    for user in users:
        if user.user_id not in user_order: continue
        i = np.where(user_order == user.user_id)[0][0]
        assert int(user.user_id) == int(user_order[i])
        assert int(user.was_correct) == int(y[i])

    print("Training Set: {}, {}".format(X_train.shape, y_train.shape))
    print("Testing Set: {}, {}".format(X_test.shape, y_test.shape))
    return X_train, X_test, np.asarray(y_train), np.asarray(y_test)


def lstm_pred(n = 0):
    users = get_users()
    # lda_analysis(users)

    top_words = 50000

    if BUILD_NEW_DATA:
        print("Retrieving data and shaping")
        users = [u for u in users if u.tweets]
        users_relevant_tweets = build_dataset(users)
        print("Subselecting best words")
        X, y, user_order = format_training_data(users_relevant_tweets)
    else:
        X,y,user_order = get_prebuilt_data()

    X, new_word_to_idx = keep_n_best_words(X,y, top_words)
    #new_idx_to_word = {idx: k for k, idx in new_word_to_idx.items()}

    #print(X[:5])
    #print([[new_idx_to_word[w] for w in x] for x in X[:5]])
    #print(Counter(y))
    # X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle=False)
    X_train, X_test, y_train, y_test  = train_test_split_on_users(X,y, user_order, users, n)
    print(Counter(y_train), Counter(y_test))

    max_tweet_length = 12
    X_train = sequence.pad_sequences(X_train, maxlen=max_tweet_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_tweet_length)

    # create the model
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

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


def main():
    global bow_corpus
    global word_to_idx, idx_to_word
    global bow_corpus_top_n
    wn.ensure_loaded()
    bow_corpus = get_corpus()

    bow_corpus_tmp = [w[0] for w in bow_corpus.items() if w[1] > 2]
    #print("Corpus size: {}".format(len(bow_corpus_tmp)))

    word_to_idx = {k: idx for idx, k in enumerate(bow_corpus_tmp)}
    idx_to_word = {idx: k for k, idx in word_to_idx.items()}

    for n in range(0,1000,100):
        lstm_pred(n)


if __name__ == "__main__":
    main()


