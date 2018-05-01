import glob, os, sys, json, datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import re, nltk
from dateutil import parser
from Fact import Fact
from User import User
from Transaction import Transaction
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)

SERVER_RUN = True

DIR = os.path.dirname(__file__) + '../../3_Data/'


def decoder(o):
    def user_decoder(obj):
        if 'user_id' not in obj.keys(): return obj
        if 'avg_time_to_retweet' in obj.keys():
            return User(obj['user_id'], tweets=obj['tweets'], fact=obj['fact'], transactions=obj['transactions'],
                        credibility=obj['credibility'],
                        controversy=obj['controversy'], features=obj['features'], was_correct=obj['was_correct'],
                        avg_time_to_retweet=obj['avg_time_to_retweet'])
        # <user_id, tweets, fact, transactions, credibility, controversy>
        return User(obj['user_id'], tweets=obj['tweets'], fact=obj['fact'], transactions=obj['transactions'],
                    credibility=obj['credibility'],
                    controversy=obj['controversy'], features=obj['features'], was_correct=obj['was_correct'])

    def fact_decoder(obj):
        # <RUMOR_TPYE, HASH, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND, SOURCE_TWEET>
        return Fact(obj['rumor_type'], obj['topic'], obj['text'], obj['true'], obj['proven_false'],
                    obj['is_turnaround'], obj['source_tweet'], hash=obj['hash'])

    def transaction_decoder(obj):
        # <sourceId, id, user_id, fact, timestamp, stance, weight>
        return Transaction(obj['sourceId'], obj['id'], obj['user_id'], obj['fact'], obj['timestamp'], obj['stance'],
                           obj['weight'])

    if 'tweets' in o.keys():
        return user_decoder(o)
    elif 'hash' in o.keys():
        return fact_decoder(o)
    elif 'sourceId' in o.keys():
        return transaction_decoder(o)
    else:
        return o


def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


def get_data():
    fact_file = glob.glob(DIR + 'facts.json')[0]
    transactions_file = glob.glob(DIR + 'factTransaction.json')[0]
    facts = json.load(open(fact_file), object_hook=decoder)
    transactions = pd.read_json(transactions_file)
    return facts, transactions


def get_users():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    print('Found {} users'.format(len(user_files)))
    if SERVER_RUN:
        user_files = sorted(user_files, reverse=False)
    else:
        user_files = sorted(user_files, reverse=True)
    if len(user_files) < 10: print('WRONG DIR?')
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=decoder)
        yield user


def write_user(user):
    print("Writing user: {}".format(user.user_id))
    with open(DIR + 'user_tweets/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')


def time_til_retweet(users, df_transactions, facts):
    print("Calculating avg time between original tweet and retweet per user")
    avg_min_to_retweet_per_user = {}
    for user in users:
        if not user.tweets or len(user.tweets) < 1: continue
        time_btw_rt = []
        if user.avg_time_to_retweet is None:
            for tweet in user.tweets:
                if not 'quoted_status' in tweet: continue
                if not 'created_at' in tweet['quoted_status']: continue
                date_original = parser.parse(tweet['quoted_status']['created_at'])
                date_retweet = parser.parse(tweet['created_at'])
                time_btw_rt.append(date_original - date_retweet)
            if len(time_btw_rt) == 0: continue

            average_timedelta = round(float((sum(time_btw_rt, datetime.timedelta(0)) / len(time_btw_rt)).seconds) / 60)
            user.avg_time_to_retweet = average_timedelta
        write_user(user)
        avg_min_to_retweet_per_user[user.user_id] = user.avg_time_to_retweet

    hist_all, bins = np.histogram(list(avg_min_to_retweet_per_user.values()))
    print(hist_all, bins)
    # plt.figure()
    # plt.hist(list(avg_min_to_retweet_per_user.values()), bins=bins)
    # # plt.show()
    # plt.figure()
    # plt.hist(list(retweet_mins_pos.values()), bins=bins)
    # # plt.show()
    # plt.figure()
    # plt.hist(list(retweet_mins_neg.values()), bins=bins)
    # # plt.show()

    fact_to_hist = {}
    X = []
    y = []
    df_transactions['time_til_retweet'] = df_transactions['user_id'].map(
        lambda uid: avg_min_to_retweet_per_user[uid] if uid in avg_min_to_retweet_per_user else np.nan)
    df_transactions.dropna(subset=['time_til_retweet'], inplace=True)
    df_grouped_transactions = df_transactions.groupby(['fact'])
    for tr_group in df_grouped_transactions:
        df_tr_g = tr_group[1]
        hist = np.histogram(df_tr_g['time_til_retweet'], bins=bins)
        fact_to_hist[tr_group[0]] = hist[0]

        # plt.figure()
        # plt.hist(list(df_tr_g['time_til_retweet']), bins=bins)
        # plt.show()
        fact = [fact for fact in facts if fact.hash == tr_group[0]][0]
        X.append(list(map(int, hist[0])))
        if fact.true == '0' or fact.true == 0:
            y.append(0)
        elif fact.true == '1' or fact.true == 1:
            y.append(1)
        else:
            y.append(-1)

    #classifier = evaluation(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = Perceptron(n_iter=50)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    match = [1 if t == p else 0 for t, p in zip(y_test, pred)]

    score = metrics.accuracy_score(y_test, pred)
    print(score)

    corclassified_group_pos = np.asarray([hist for hist, m, y in zip(X_test, match, y_test) if m == 1 and y == 1])
    corclassified_group_neg = np.asarray([hist for hist, m, y in zip(X_test, match, y_test) if m == 1 and y == 0])
    corclassified_group_unk = np.asarray([hist for hist, m, y in zip(X_test, match, y_test) if m == 1 and y == -1])

    misclassified_group_pos = np.asarray([hist for hist, m, y in zip(X_test, match, y_test) if m == 0 and y == 1])
    misclassified_group_neg = np.asarray([hist for hist, m, y in zip(X_test, match, y_test) if m == 0 and y == 0])
    misclassified_group_unk = np.asarray([hist for hist, m, y in zip(X_test, match, y_test) if m == 0 and y == -1])

    corclass_pos_avg = corclassified_group_pos.mean(0)
    corclass_neg_avg = corclassified_group_neg.mean(0)
    corclass_unk_avg = corclassified_group_unk.mean(0)
    misclass_pos_avg = misclassified_group_pos.mean(0)
    misclass_neg_avg = misclassified_group_neg.mean(0)
    misclass_unk_avg = misclassified_group_unk.mean(0)

    fig, axes = plt.subplots(2, 3)
    axes[0, 0].bar(bins[:-1], corclass_pos_avg, width=np.diff(bins), ec="k", align="edge")
    axes[0, 0].set_title('Correct classified where fact was true')
    axes[0, 1].bar(bins[:-1], corclass_neg_avg, width=np.diff(bins), ec="k", align="edge")
    axes[0, 1].set_title('Correct classified where fact was false')
    axes[0, 2].bar(bins[:-1], corclass_unk_avg, width=np.diff(bins), ec="k", align="edge")
    axes[0, 2].set_title('Correct classified where fact was unknown')

    axes[1, 0].bar(bins[:-1], misclass_pos_avg, width=np.diff(bins), ec="r", align="edge")
    axes[1, 0].set_title('Incorrect classified where fact was true')
    axes[1, 1].bar(bins[:-1], misclass_neg_avg, width=np.diff(bins), ec="r", align="edge")
    axes[1, 1].set_title('Incorrect classified where fact was false')
    axes[1, 2].bar(bins[:-1], misclass_unk_avg, width=np.diff(bins), ec="r", align="edge")
    axes[1, 2].set_title('Incorrect classified where fact was unknown')
    #plt.show()

    return df_transactions


def evaluation(X, y):
    def benchmark(clf):
        # print('_' * 80)
        # print("Training: ")
        # print(clf)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        scores = cross_val_score(clf, X_test, y_test, cv=5)

        score = metrics.accuracy_score(y_test, pred)
        # print("accuracy:   %0.3f" % score)
        print("Cross validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores.mean()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest"),
            (MultinomialNB(alpha=.01), "Multinomial NB"),
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


def main():
    facts, df_transactions = get_data()
    users = get_users()
    # feature: histogram of avg time of retweet per user. Each user is one count in the histogram. Each rumor has one histogram.
    # Prediction with this feature: ~.6 acc, high variance
    df_transactions = time_til_retweet(users, df_transactions, facts)



if __name__ == "__main__":
    main()
