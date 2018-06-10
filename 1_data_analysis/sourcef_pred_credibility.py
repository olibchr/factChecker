from __future__ import print_function
import glob
import json
import multiprocessing
import os
import pickle
import sys
import warnings
from string import digits
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.covariance import EllipticEnvelope

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, normalize, StandardScaler
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import TruncatedSVD, PCA
import seaborn as sns

sns.set(style="ticks")

sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
sys.path.insert(0, os.path.dirname(__file__) + '../5_models')
from decoder import decoder
from metrics import ndcg_score

warnings.filterwarnings("ignore", category=DeprecationWarning)
# fix random seed for reproducibility
BUILD_NEW_DATA = False

DIR = os.path.dirname(__file__) + '../../3_Data/'
num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))
fact_to_words = {}
word_vectors = 0  # KeyedVectors.load_word2vec_format('model_data/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')


def tokenize_text(text, only_retweets=False):
    tokenizer = RegexpTokenizer(r'\w+')
    links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for l in links: text = text.replace(l, ' evidence ')
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


def get_relevant_tweets(user):
    relevant_tweets = []
    print(user.user_id)
    user_fact_words = fact_to_words[user.fact]
    for tweet in user.tweets:
        distance_to_topic = []
        tokens = tokenize_text(tweet['text'], only_retweets=False)
        for token in tokens:
            if token not in word_vectors.vocab: continue
            increment = np.average(word_vectors.distances(token, other_words=user_fact_words))
            distance_to_topic.append(increment)
        if np.average(np.asarray(distance_to_topic)) < 0.8:
            relevant_tweets.append(tweet)

    return relevant_tweets


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


def build_features_for_user(user):
    # Message based features: number of swear language words
    # Source based features length of screen name, has URL, ratio of followers to followees

    avg_len = []
    avg_words = []
    all_characters = []
    avg_unique_char = []
    retweets = []
    tweets_that_are_retweet = []
    avg_special_symbol = []
    avg_emoticons = []
    tweets_that_are_reply = []
    avg_links = []
    avg_hashtags = []
    avg_mentions = []
    avg_questionM = []
    avg_exlamationM = []
    avg_multiQueExlM = []
    avg_upperCase = []
    avg_sent_pos = []
    avg_sent_neg = []
    avg_count_distinct_hashtags = []
    most_common_weekday = []
    most_common_hour = []
    avg_search_results = []
    avg_search_r_is_news_page = []
    avg_count_distinct_words = []


    emoji_pattern = re.compile(
        '^(:\(|:\))+$'
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)

    link_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    relevant_tweets = user.tweets  # get_relevant_tweets(user)
    for t in relevant_tweets:
        avg_len.append(len(t['text']))
        avg_words.append(len(tokenize_text(t['text'])))
        chars = [c for c in t['text']]
        all_characters.extend(chars)
        avg_unique_char.append(len(set(chars)))
        avg_hashtags.append(len([c for c in chars if c == '#']))
        avg_mentions.append(len([c for c in chars if c == '@']))
        retweets.append(int(t['retweets']) if 'retweets' in t else 0)
        if 'quoted_status' in t and t['quoted_status'] is not None: tweets_that_are_retweet.append(t)
        avg_special_symbol.append(len(re.findall('[^0-9a-zA-Z *]', t['text'])))
        avg_emoticons.append(1 if emoji_pattern.search(t['text']) is not None else 0)
        if 'reply' in t and t['reply'] is not None: tweets_that_are_reply.append(t)
        avg_links.append(1 if link_pattern.search(t['text']) is not None else 0)
    if len(relevant_tweets) == 0: relevant_tweets = [0]; retweets = [0]

    avg_len = 1.0 * sum(avg_len) / len(relevant_tweets)
    avg_words = 1.0 * sum(avg_words) / len(relevant_tweets)
    avg_unique_char = 1.0 * sum(avg_unique_char) / len(relevant_tweets)
    avg_hashtags = sum(avg_hashtags) / len(relevant_tweets)
    avg_retweets = 1.0 * sum(retweets) / len(retweets)
    pos_words = int(user.features['pos_words']) if 'pos_words' in user.features else 0
    neg_words = int(user.features['neg_words']) if 'neg_words' in user.features else 0
    avg_tweet_is_retweet = len(tweets_that_are_retweet) / len(relevant_tweets)
    avg_special_symbol = sum(avg_special_symbol) / len(relevant_tweets)
    avg_emoticons = 1.0 * sum(avg_emoticons) / len(relevant_tweets)
    avg_tweet_is_reply = len(tweets_that_are_reply) / len(relevant_tweets)
    avg_mentions = sum(avg_mentions) / len(relevant_tweets)
    avg_links = 1.0 * sum(avg_links) / len(relevant_tweets)

    # followers, friends, description, created_at, verified, statuses_count, lang}
    followers = int(user.features['followers']) if 'followers' in user.features else 0
    friends = int(user.features['friends']) if 'friends' in user.features else 0
    verified = 1 if 'verified' in user.features and user.features['verified'] == 'true' else 0
    status_cnt = int(user.features['statuses_count']) if 'statuses_count' in user.features else 0
    time_retweet = int(user.avg_time_to_retweet)
    len_description = len(user.features['description']) if 'description' in user.features else 0
    len_name = len(user.features['name']) if 'name' in user.features else 0

    return {
        'user_id': user.user_id,
        'y': user.was_correct,
        'avg_len': avg_len,
        'avg_words': avg_words,
        'avg_unique_char': avg_unique_char,
        'avg_hashtags': avg_hashtags,
        'avg_retweets': avg_retweets,
        'pos_words': pos_words,
        'neg_words': neg_words,
        'avg_tweet_is_retweet': avg_tweet_is_retweet,
        'avg_special_symbol': avg_special_symbol,
        'avg_emoticons': avg_emoticons,
        'avg_tweet_is_reply': avg_tweet_is_reply,
        'avg_mentions': avg_mentions,
        'avg_links': avg_links,
        'followers': followers,
        'friends': friends,
        'verified': verified,
        'status_cnt': status_cnt,
        'time_retweet': time_retweet,
        'len_description': len_description,
        'len_name': len_name
    }


def benchmark(clf, X_train, X_test, y_train, y_test, X, y):
    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross validation: {}".format(np.average(scores)))

    clf.fit(X=X_train, y=y_train)
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    precision, recall, fscore, sup = precision_recall_fscore_support(y_test, pred, average='macro')
    ndgc = ndcg_score(y_test, pred)
    print("NDCG: {}".format(ndgc))
    print("CLF score: {}".format(clf.score(X_test, y_test)))
    print("Accuracy: %0.3f, Precision: %0.3f, Recall: %0.3f, F1 score: %0.3f" % (
        score, precision, recall, fscore))


def evaluation(X, y, X_train=None, X_test=None, y_train=None, y_test=None):
    def benchmark(clf):
        scores = cross_val_score(clf, X, y, cv=5)

        clf.fit(X_train_imp, y_train)
        pred = clf.predict(X_test_imp)

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
        return fscore, fscore2, scores.mean()

    print('&' * 80)
    # print("Evaluation")

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

    # Train sparse SVM
    for penalty in ["l2", "l1"]:
        svms = [
            #(LinearSVC(penalty=penalty, dual=False, tol=1e-3), "Linear SVM"),
            #(SVC(kernel='rbf', degree=2), "RBF SVC"),
            #(SVC(kernel='sigmoid', degree=2), "Sigm SVC"),
            (SVC(C=1, gamma=1), "Best param SVC)"),
            #(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty), "SGDC")
        ]
        print("%s penalty" % penalty.upper())
        for clf, name in svms:
            # print('=' * 80)
            print(name)
            results.append([benchmark(clf)])
    return results


def reject_outliers(data, y, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m], y[s < m]


def model_param_grid_search(X, y):
    from matplotlib.colors import Normalize
    class MidpointNormalize(Normalize):

        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    svd = TruncatedSVD(2)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_2d = lsa.fit_transform(X, y)

    print("Do some magic")
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    C_2d_range = [1e-2, 1, 1e2]
    gamma_2d_range = [1e-1, 1, 1e1]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X_2d, y)
            classifiers.append((C, gamma, clf))
    print("Do some more magic")
    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # visualize decision function for these parameters
        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
                  size='medium')

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.RdBu_r,
                    edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()
    pass


def sourcef_pred(chi_k=15, ldak=5):
    # Utility function to move the midpoint of a colormap to be around
    # the values of interest.
    global bow_corpus
    global word_to_idx
    global fact_to_words
    wn.ensure_loaded()

    if BUILD_NEW_DATA:
        users = get_users()
        print("Getting user features")
        fact_topics = build_fact_topics()
        # fact_to_words = {r['hash']: [w for w in r['fact_terms'] if w in word_vectors.vocab] for index, r in fact_topics[['hash', 'fact_terms']].iterrows()}
        users_with_tweets = [u for u in users if len(u.tweets) > 0]
        users_with_features = Parallel(n_jobs=num_jobs)(
            delayed(build_features_for_user)(user) for i, user in enumerate(users_with_tweets))
        with open('model_data/user_featues', 'wb') as tmpfile:
            pickle.dump(users_with_features, tmpfile)
    else:
        with open('model_data/user_featues', 'rb') as tmpfile:
            users_with_features = pickle.load(tmpfile)
    users_df = pd.DataFrame(users_with_features)

    # print(users_df.describe())

    features = ['avg_len', 'avg_words', 'avg_unique_char', 'avg_hashtags', 'avg_retweets', 'pos_words', 'neg_words',
                'avg_tweet_is_retweet', 'avg_special_symbol', 'avg_mentions',
                'avg_links', 'followers', 'friends', 'status_cnt', 'time_retweet', 'len_description',
                'len_name']
    X = users_df[features].values
    y = users_df['y'].values

    #  %%%%%% Correlation plot %%%%%%
    corr = users_df[features].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.savefig('foo.png')

    ada = RandomOverSampler(random_state=42)
    X, y = ada.fit_sample(X, y)

    # Remove outliers
    # env = EllipticEnvelope().fit(X, y)
    # outliers = env.predict(X)
    # X = X[outliers != -1]
    # y = y[outliers != -1]

    svd = TruncatedSVD(2)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd)
    X_2d = lsa.fit_transform(X, y)
    X_2d = normalize(X_2d, axis=0)

    # 2d plot of X
    X2d_df = pd.DataFrame({'x1': X_2d[:, 0], 'x2': X_2d[:, 1], 'y': y})
    # sns.lmplot(data=X2d_df, x='x1', y='x2', hue='y')
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    ch2 = SelectKBest(chi2, k=chi_k)
    X = ch2.fit_transform(X, y)
    X_train = ch2.transform(X_train)
    X_test = ch2.transform(X_test)
    scaler = StandardScaler()
    X = scaler.fit_transform(X, y)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    evaluation(X, y, X_train, X_test, y_train, y_test)


def main():
    sourcef_pred(15, 10)



if __name__ == "__main__":
    main()
