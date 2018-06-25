from __future__ import print_function

import datetime
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
from dateutil import parser
from gensim.models import KeyedVectors
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, normalize, StandardScaler
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation
import seaborn as sns

sns.set(style="ticks")

sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
sys.path.insert(0, os.path.dirname(__file__) + '../5_fact_checking_models')
from decoder import decoder
from metrics import ndcg_score

warnings.filterwarnings("ignore", category=DeprecationWarning)
# fix random seed for reproducibility
BUILD_NEW_DATA = False
LDA_TOPIC = False
NEW_LDA_MODEL = False

DIR = os.path.dirname(__file__) + '../../3_Data/'
num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))
fact_to_words = {}
lda = ()
users = ()
lda_text_to_id = {}
lda_topics_per_text = []
word_vectors = KeyedVectors.load_word2vec_format('model_data/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')
sid = SentimentIntensityAnalyzer()


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


def get_relevant_tweets(user, i=0.8):
    def topic_overlap(t1, t2):
        n_topics = 5
        threshold = 2
        t1_topics = lda_topics_per_text[lda_text_to_id[t1]]
        t2_topics = lda_topics_per_text[lda_text_to_id[t2]]
        t_topics1 = t1_topics.argsort()[-n_topics:][::-1]
        t_topics2 = t2_topics.argsort()[-n_topics:][::-1]
        overlap = [val for val in t_topics1 if val in t_topics2]
        if len(overlap) >= threshold:
            return True
        return False

    relevant_tweets = []
    # print(user.user_id)
    user_fact_words = fact_to_words[user.fact]
    for tweet in user.tweets:
        distance_to_topic = []
        tokens = tokenize_text(tweet['text'], only_retweets=False)
        if LDA_TOPIC:
            if topic_overlap(tweet['text'], ' '.join(user_fact_words)):
                relevant_tweets.append(tweet)
            continue
        for token in tokens:
            if token not in word_vectors.vocab: continue
            increment = np.average(word_vectors.distances(token, other_words=[ufw for ufw in user_fact_words if
                                                                              ufw in word_vectors.vocab]))
            distance_to_topic.append(increment)
        if np.average(np.asarray(distance_to_topic)) < i:
            relevant_tweets.append(tweet)
    return relevant_tweets


def lda_analysis(users):
    global lda_text_to_id, lda_topics_per_text

    n_features = 1000
    n_components = 50
    n_top_words = 10
    print("Constructing user docs")
    X = [[tweet['text'] for tweet in user.tweets] for user in users]
    X = [tweet for sublist in X for tweet in sublist]
    fact_topics = build_fact_topics()

    for t in [' '.join(f) for f in fact_topics['fact_terms'].values]: X.append(t)

    print(X[:5])
    print("TF fitting user docs")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    tf = tf_vectorizer.fit(X)
    X_tf = tf.transform(X)

    if NEW_LDA_MODEL:
        print("Training new LDA model")
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(X_tf)
        with open('model_data/lda_model', 'wb') as tmpfile:
            pickle.dump(lda, tmpfile)
    else:
        with open('model_data/lda_model', 'rb') as tmpfile:
            lda = pickle.load(tmpfile)

    lda_text_to_id = {txt: id for id, txt in enumerate(X)}
    lda_topics_per_text = lda.transform(X_tf)

    tf_feature_names = tf_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([tf_feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    return lda


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


def build_features_for_user(user, i=0.8):
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
    avg_tweets_on_this_topic = []
    avg_personal_pronoun_first = []

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

    relevant_tweets = get_relevant_tweets(user, i)
    for t in relevant_tweets:
        avg_len.append(len(t['text']))
        tokenized_text = tokenize_text(t['text'])
        if len(tokenized_text) == 0: tokenized_text = [0]
        avg_words.append(len(tokenized_text))
        avg_count_distinct_words.extend(tokenized_text)
        chars = [c for c in t['text']]
        all_characters.extend(chars)
        avg_unique_char.append(len(set(chars)))
        avg_hashtags.append(len([c for c in chars if c == '#']))
        avg_mentions.append(len([c for c in chars if c == '@']))
        retweets.append(int(t['retweets']) if 'retweets' in t else 0)
        if 'quoted_status' in t and t['quoted_status'] is not None: tweets_that_are_retweet.append(t)
        avg_special_symbol.append(len(re.findall('[^0-9a-zA-Z *]', t['text'])))
        avg_questionM.append(1 if '?' in t['text'] else 0)
        avg_exlamationM.append(1 if '!' in t['text'] else 0)
        avg_multiQueExlM.append(
            1 if len(re.findall('/[?]/', t['text'])) + len(re.findall('/[!]/', t['text'])) > 1 else 0)
        avg_upperCase.append(len(re.findall('/[A-Z]/', t['text'])))
        avg_personal_pronoun_first.append(1 if tokenized_text[0] in pronouns else 0)

        avg_sent_pos.append(sid.polarity_scores(t['text'])['pos'])
        avg_sent_neg.append(sid.polarity_scores(t['text'])['neg'])
        avg_count_distinct_hashtags.append((len(re.findall('/[#]/', t['text']))))

        timestamp = parser.parse(t['created_at'])
        most_common_weekday.append(timestamp.day)
        most_common_hour.append(timestamp.hour)
        avg_emoticons.append(len(re.findall(emoji_pattern, t['text'])))
        if 'reply' in t and t['reply_to'] is not None: tweets_that_are_reply.append(t)
        avg_links.append(len(re.findall(link_pattern, t['text'])))
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

    avg_questionM = 1.0 * sum(avg_questionM) / len(relevant_tweets)
    avg_exlamationM = 1.0 * sum(avg_exlamationM) / len(relevant_tweets)
    avg_multiQueExlM = 1.0 * sum(avg_multiQueExlM) / len(relevant_tweets)
    avg_upperCase = 1.0 * sum(avg_upperCase) / len(relevant_tweets)
    avg_sent_pos = 1.0 * sum(avg_sent_pos) / len(relevant_tweets)
    avg_sent_neg = 1.0 * sum(avg_sent_neg) / len(relevant_tweets)
    avg_count_distinct_hashtags = 1.0 * sum(avg_count_distinct_hashtags) / len(relevant_tweets)
    most_common_weekday = 1.0 * sum(most_common_weekday) / len(relevant_tweets)
    most_common_hour = 1.0 * sum(most_common_hour) / len(relevant_tweets)
    avg_count_distinct_words = 1.0 * len(set(avg_count_distinct_words)) / len(relevant_tweets)
    avg_tweets_on_this_topic = len(relevant_tweets) * 1.0 / len(user.tweets)
    avg_personal_pronoun_first = sum(avg_personal_pronoun_first) * 1.0 / len(user.tweets)

    # followers, friends, description, created_at, verified, statuses_count, lang}
    reg_age = int((datetime.datetime.now().replace(tzinfo=None) - parser.parse(user.features['created_at']).replace(
        tzinfo=None)).days if 'created_at' in user.features else 0)
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
        'avg_questionM': avg_questionM,
        'avg_exlamationM': avg_exlamationM,
        'avg_multiQueExlM': avg_multiQueExlM,
        'avg_personal_pronoun_first': avg_personal_pronoun_first,
        'avg_upperCase': avg_upperCase,
        'avg_sent_pos': avg_sent_pos,
        'avg_sent_neg': avg_sent_neg,
        'avg_count_distinct_hashtags': avg_count_distinct_hashtags,
        'most_common_weekday': most_common_weekday,
        'most_common_hour': most_common_hour,
        'avg_count_distinct_words': avg_count_distinct_words,
        'avg_tweets_on_this_topic': avg_tweets_on_this_topic,
        'followers': followers,
        'friends': friends,
        'verified': verified,
        'status_cnt': status_cnt,
        'time_retweet': time_retweet,
        'len_description': len_description,
        'len_name': len_name,
        'reg_age': reg_age
    }


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


def sourcef_pred(chi_k=15, ldak=5, proximity=0.8):
    # Utility function to move the midpoint of a colormap to be around
    # the values of interest.
    global bow_corpus
    global word_to_idx
    global fact_to_words
    global lda
    wn.ensure_loaded()

    print(chi_k, ldak)

    if BUILD_NEW_DATA:
        users = get_users()
        if LDA_TOPIC: lda = lda_analysis(users)
        print("Getting user features")
        fact_topics = build_fact_topics()
        fact_to_words = {r['hash']: [w for w in r['fact_terms']] for index, r in
                         fact_topics[['hash', 'fact_terms']].iterrows()}
        users_with_tweets = [u for u in users if len(u.tweets) > 0]
        users_with_features = Parallel(n_jobs=num_jobs)(
            delayed(build_features_for_user)(user, proximity) for i, user in enumerate(users_with_tweets))
        with open('model_data/user_features.pkl', 'wb') as tmpfile:
            pickle.dump(users_with_features, tmpfile)
    else:
        with open('model_data/user_features.pkl', 'rb') as tmpfile:
            users_with_features = pickle.load(tmpfile)
    users_df = pd.DataFrame(users_with_features)

    features = ['avg_len', 'avg_words', 'avg_unique_char', 'avg_hashtags', 'avg_retweets', 'pos_words', 'neg_words',
                'avg_tweet_is_retweet', 'avg_special_symbol', 'avg_mentions', 'avg_emoticons',
                'avg_questionM', 'avg_exlamationM', 'avg_sent_pos',
                'avg_links', 'followers', 'friends', 'status_cnt', 'time_retweet', 'len_description', 'len_name',
                'avg_sent_neg', 'avg_count_distinct_words', 'avg_tweets_on_this_topic', 'avg_multiQueExlM', 'reg_age',
                'avg_upperCase', 'avg_count_distinct_hashtags', 'avg_tweet_is_reply', 'avg_personal_pronoun_first'
                ]

    #  %%%%%% Correlation plot %%%%%%
    corr = users_df[list(features)].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.show()

    features = ['avg_words', 'avg_retweets', 'avg_tweet_is_retweet', 'avg_special_symbol', 'avg_emoticons', 'avg_links',
                'friends', 'time_retweet', 'len_description', 'avg_sent_neg', 'avg_count_distinct_words',
                'avg_personal_pronoun_first', 'followers', 'len_name',
                ]

    X = users_df[list(features)].values
    y = users_df['y'].values

    ada = RandomOverSampler(random_state=42)
    X, y = ada.fit_sample(X, y)

    lsa = make_pipeline(StandardScaler(), TruncatedSVD(2))
    X_2d = lsa.fit_transform(X, y)
    X_2d = normalize(X_2d, axis=0)
    # 2d plot of X
    X2d_df = pd.DataFrame({'x1': X_2d[:, 0], 'x2': X_2d[:, 1], 'y': y})
    sns.lmplot(data=X2d_df, x='x1', y='x2', hue='y')
    # plt.show()

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i in range(1, len(features) + 1):
        ax = fig.add_subplot(3, 5, i)
        sns.boxplot(x="y", y=features[i - 1], data=users_df, palette="Set3")
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=ldak), SVC(C=1, gamma=1))
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

    # return evaluation(X, y, X_train, X_test, y_train, y_test)


def main():
    # for i in np.arange(10, 25):
    sourcef_pred(-1, 10)


if __name__ == "__main__":
    main()
