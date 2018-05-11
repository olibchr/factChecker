import glob, os, sys, json, datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import re, nltk
from dateutil import parser
from Fact import Fact
from User import User
from Transaction import Transaction
from decoder import decoder

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
from sklearn.preprocessing import Imputer
import warnings
import scipy.stats
from collections import Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)

SERVER_RUN = True

DIR = os.path.dirname(__file__) + '../../3_Data/'


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

    avg_min_to_retweet_per_user = {user.user_id: user.avg_time_to_retweet for user in users if user.avg_time_to_retweet is not None}
    avg_sent_per_user = {user.sent_tweets_avg: user.sent_tweets_avg for user in users if user.sent_tweets_avg is not None}

    print(avg_sent_per_user)
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

    X = []
    y = []
    df_transactions['time_til_retweet'] = df_transactions['user_id'].map(
        lambda uid: avg_min_to_retweet_per_user[uid] if uid in avg_min_to_retweet_per_user else np.nan)
    df_transactions['user_sent'] = df_transactions['user_id'].map(
        lambda uid: avg_sent_per_user[uid] if uid in avg_sent_per_user else np.nan)

    df_transactions.dropna(subset=['time_til_retweet'], inplace=True)
    #df_transactions.dropna(subset=['user_sent'], inplace=True)

    df_grouped_transactions = df_transactions.groupby(['fact'])
    for tr_group in df_grouped_transactions:
        df_tr_g = tr_group[1]
        #hist = np.histogram(df_tr_g['time_til_retweet'], bins=bins)
        #print(df_tr_g['time_til_retweet'])

        indices = ~np.isnan(df_tr_g['time_til_retweet'])
        # print(indices)
        #if len(indices) == 0 or df_tr_g['time_til_retweet'][indices].shape[0] == 0: continue
        time_til_retweet_avg = np.average(df_tr_g['time_til_retweet'][indices])
        time_til_retweet_var = np.var(df_tr_g['time_til_retweet'][indices])
        time_til_retweet_mode = np.asarray(scipy.stats.mode(df_tr_g['time_til_retweet'][indices]))[0][0]

        fact = [fact for fact in facts if fact.hash == tr_group[0]][0]
        if fact.true == '0' or fact.true == 0:
            y.append(0)
        elif fact.true == '1' or fact.true == 1:
            y.append(1)
        else:
            continue
            y.append(-1)
        X.append(np.asarray([time_til_retweet_avg, time_til_retweet_var, time_til_retweet_mode]))

    classifier = evaluation(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = Perceptron(n_iter=50)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    match = [1 if t == p else 0 for t, p in zip(y_test, pred)]

    score = metrics.accuracy_score(y_test, pred)
    print(score)
    print(len(X), len(X_train), len(X_test))

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
    print(corclassified_group_pos)
    print(corclassified_group_neg)
    print(corclassified_group_unk)
    print(misclassified_group_pos)
    print(misclassified_group_neg)
    print(misclassified_group_unk)

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
        clf.fit(X_train_imp, y_train)

        pred = clf.predict(X_test_imp)
        scores = cross_val_score(clf, X_test_imp, y_test, cv=5)

        score = metrics.accuracy_score(y_test, pred)
        # print("accuracy:   %0.3f" % score)
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
