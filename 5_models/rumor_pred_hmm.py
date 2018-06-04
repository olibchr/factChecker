import pandas as pd
from hmmlearn import hmm
import sklearn.metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
import random
from datetime import datetime, timedelta
import sys, glob, os
sys.path.insert(0, os.path.dirname(__file__) + '../0_data_retrieval')
sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
import time
import numpy as np
import warnings
from collections import Counter
from data_extraction_pheme import time_multi_plot
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore", category=DeprecationWarning)

DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/RAW/'
SHOW_PLOTS = False
HMM_COMPONENTS = [7, 7, 5]


def get_data():
    facts = pd.read_json(DIR + '../facts.json')
    transactions = pd.read_json(DIR + '../factTransaction.json')
    return facts, transactions


def wisdom_crowd(facts, transactions):
    print('Prediction using Wisdom of the Crowd')
    # Prediction: True = 1; False = 0
    pred = []

    # If its confirmed true, but later proven false, then it should not be true
    facts.loc[facts['proven_false'] == 1, 'true'] = 0
    truth = []
    for i, f in facts.iterrows():
        if f['true'] == '0':
            truth.append(0)
        elif f['true'] == '1':
            truth.append(1)
        else:
            truth.append(-1)
        corr_transactions = transactions[(transactions['fact'] == f['hash'])]
        support = len([s for i, s in corr_transactions.iterrows() if s['stance'] == 'supporting'])
        reject = len([r for i, r in corr_transactions.iterrows() if r['stance'] == 'denying'])
        if support > reject:
            pred.append(1)
        elif reject > support:
            pred.append(0)
        else:
            pred.append(-1)

    match = [1 if t == p else 0 for t, p in zip(truth, pred)]
    print('Predicting 1: {}'.format(len([s for s in pred if s == 1])))
    print('Accuracy: {}, Correct: {}, Total: {}'.format(sum(match) / (len(match) * 1.0), sum(match), len(match)))

    f1_score = sklearn.metrics.f1_score(truth, pred, average='micro')
    print('F1 Score: {}'.format(f1_score))
    print('Categories: {}'.format(Counter(facts['true'])))
    print('Predicted Categories: {}'.format(Counter(pred)))
    print('Missed facts with labels: {}'.format(Counter([facts['true'][idx] for idx, m in enumerate(match) if m == 0])))
    print('Missed predictions with labels: {}'.format(Counter([pred[idx] for idx, m in enumerate(match) if m == 0])))
    misclassified_hash = [hsh for hsh, m in zip(facts.as_matrix(['hash']).flatten(), match) if m == 0]
    corclassified_hash = [hsh for hsh, m in zip(facts.as_matrix(['hash']).flatten(), match) if m == 1]

    # print('$$$$ Detailed Analyiss $$$$')
    # print('\tCorrectly classified')
    # classification_analysis(facts, transactions, corclassified_hash)
    # print('\tIncorrectly classified')
    # classification_analysis(facts, transactions, misclassified_hash)
    precision_recall(truth, pred, 2)
    aggregated_analysis(facts, transactions, Counter(misclassified_hash), Counter(corclassified_hash))


def wisdom_certainty(facts, transactions):
    print('Prediction using Weighted Wisdom of the Crowd')
    # Prediction: True = 1; False = 0
    pred = []

    # If its confirmed true, but later proven false, then it should not be true
    facts.loc[facts['proven_false'] == 1, 'true'] = 0

    transactions.loc[transactions['weight'] == 'certain', 'weight'] = 3
    transactions.loc[transactions['weight'] == 'somewhat-certain', 'weight'] = 2
    transactions.loc[transactions['weight'] == 'uncertain', 'weight'] = 1
    transactions.loc[transactions['weight'] == 'underspecified', 'weight'] = random.uniform(1, 3)

    truth = []
    for i, f in facts.iterrows():
        if f['true'] == '0':
            truth.append(0)
        elif f['true'] == '1':
            truth.append(1)
        else:
            truth.append(-1)
        corr_transactions = transactions[(transactions['fact'] == f['hash'])]
        support = [s for i, s in corr_transactions.iterrows() if s['stance'] == 'supporting']
        reject = [r for i, r in corr_transactions.iterrows() if r['stance'] == 'denying']

        support_weighted = sum([s['weight'] for s in support])
        reject_weighted = sum([r['weight'] for r in reject])

        if support_weighted > reject_weighted:
            pred.append(1)
        elif reject_weighted > support_weighted:
            pred.append(0)
        else:
            pred.append(-1)

    match = [1 if t == p else 0 for t, p in zip(truth, pred)]
    print('Predicting 1: {}'.format(len([s for s in pred if s == 1])))
    print('Accuracy: {}, Correct: {}, Total: {}'.format(sum(match) / (len(match) * 1.0), sum(match), len(match)))

    f1_score = sklearn.metrics.f1_score(truth, pred, average='micro')
    print('F1 Score: {}'.format(f1_score))
    print('Categories: {}'.format(Counter(facts['true'])))
    print('Predicted Categories: {}'.format(Counter(pred)))
    print('Missed facts with labels: {}'.format(Counter([facts['true'][idx] for idx, m in enumerate(match) if m == 0])))
    print('Missed predictions with labels: {}'.format(Counter([pred[idx] for idx, m in enumerate(match) if m == 0])))

    misclassified_hash = [hsh for hsh, m in zip(facts.as_matrix(['hash']).flatten(), match) if m == 0]
    corclassified_hash = [hsh for hsh, m in zip(facts.as_matrix(['hash']).flatten(), match) if m == 1]

    # print('$$$$ Detailed Analyiss $$$$')
    # print('\tCorrectly classified')
    # classification_analysis(facts, transactions, corclassified_hash)
    # print('\tIncorrectly classified')
    # classification_analysis(facts, transactions, misclassified_hash)
    precision_recall(truth, pred, 2)
    aggregated_analysis(facts, transactions, Counter(misclassified_hash), Counter(corclassified_hash))


def hmm_prediction(facts, transactions):
    print('\n%%%%%%%%%%%%%%%%%%%%')
    print('Prediction using HMM')

    # Preprocessing
    hmm_transactions = transactions[['fact', 'stance', 'weight', 'timestamp']].sort_values(by=['fact', 'timestamp'])
    facts = facts.sample(frac=1)
    facts_train = facts[:100].sort_values(by=['hash'])
    facts_test = facts[100:].sort_values(by=['hash'])

    hmm_transactions = hmm_transactions[hmm_transactions.stance != 'appeal-for-more-information']
    hmm_transactions = hmm_transactions[hmm_transactions.stance != 'comment']
    hmm_transactions = hmm_transactions[hmm_transactions.stance != 'underspecified']

    hmm_transactions.loc[hmm_transactions['weight'] == 'certain', 'weight'] = 3.0
    hmm_transactions.loc[hmm_transactions['weight'] == 'somewhat-certain', 'weight'] = 2.0
    hmm_transactions.loc[hmm_transactions['weight'] == 'uncertain', 'weight'] = 1.0
    hmm_transactions.loc[hmm_transactions['weight'] == 'underspecified', 'weight'] = 0.0

    hmm_transactions.loc[hmm_transactions['stance'] == 'supporting', 'stance'] = 1.0
    hmm_transactions.loc[hmm_transactions['stance'] == 'denying', 'stance'] = 0.0

    for t in hmm_transactions.iterrows():
        hmm_transactions.loc[t[0], 'weekday'] = t[1][3].to_datetime().weekday()
        hmm_transactions.loc[t[0], 'hour_of_day'] = t[1][3].to_datetime().hour

    def to_stamp(x):
        return time.mktime(x.to_datetime().timetuple())

    hmm_transactions['timestamp'] = hmm_transactions['timestamp'].apply(to_stamp)

    for hsh in facts.as_matrix(['hash']).flatten():
        fact_transactions = hmm_transactions[hmm_transactions.fact == hsh]
        min_ts = fact_transactions.as_matrix(['timestamp']).flatten()[0]
        for t in fact_transactions.iterrows():
            hmm_transactions.loc[t[0], 'since_start'] = t[1][3] - min_ts

    hmm_transactions[['since_start']] /= hmm_transactions[['since_start']].max()
    hmm_transactions[['weekday']] /= hmm_transactions[['weekday']].max()
    hmm_transactions[['hour_of_day']] /= hmm_transactions[['hour_of_day']].max()

    # Split data set
    facts_true = facts_train[(facts_train.true.astype(str) == '1')]
    facts_false = facts_train[(facts_train.true.astype(str) == '0')]
    facts_unknown = facts_train[(facts_train.true.astype(str) != '0') & (facts_train.true.astype(str) != '1')]

    X_true = hmm_transactions[hmm_transactions.fact.isin(facts_true['hash'])].as_matrix(
        ['stance', 'weight', 'timestamp', 'weekday', 'hour_of_day'])
    X_false = hmm_transactions[hmm_transactions.fact.isin(facts_false['hash'])].as_matrix(
        ['stance', 'weight', 'timestamp', 'weekday', 'hour_of_day'])
    X_unknown = hmm_transactions[hmm_transactions.fact.isin(facts_unknown['hash'])].as_matrix(
        ['stance', 'weight', 'timestamp', 'weekday', 'hour_of_day'])

    lengths_t = [len(np.where(hmm_transactions.fact == f)[0]) for f in sorted(facts_true.as_matrix(['hash']).flatten())]
    lengths_f = [len(np.where(hmm_transactions.fact == f)[0]) for f in
                 sorted(facts_false.as_matrix(['hash']).flatten())]
    lengths_u = [len(np.where(hmm_transactions.fact == f)[0]) for f in
                 sorted(facts_unknown.as_matrix(['hash']).flatten())]

    # Train it

    model_t = hmm.GaussianHMM(n_components=HMM_COMPONENTS[0], covariance_type="diag").fit(X_true, lengths_t)
    model_f = hmm.GaussianHMM(n_components=HMM_COMPONENTS[1], covariance_type="diag").fit(X_false, lengths_f)
    model_u = hmm.GaussianHMM(n_components=HMM_COMPONENTS[2], covariance_type="diag").fit(X_unknown, lengths_u)

    # Scoring
    pred = []
    iter_pred = []
    truth = []
    confidence = []
    confidence_all_classes = []
    for f in facts_test.as_matrix(['hash']).flatten():
        y = str(facts_test[facts_test.hash == f].as_matrix(['true']).flatten()[0])
        # if y != '0' or y != '1': continue
        if y == '0':
            truth.append(0)
        elif y == '1':
            truth.append(1)
        else:
            truth.append(-1)

        t = hmm_transactions[hmm_transactions.fact == f].as_matrix(
            ['stance', 'weight', 'timestamp', 'weekday', 'hour_of_day'])
        t_pred = []
        t_conf = []
        if len(t) == 1:
            for i in range(10):
                t = [list(map(float, tr)) for tr in t]
                log_t = model_t.score(t)
                log_f = model_f.score(t)
                log_u = model_u.score(t)

                if log_t > log_f and log_t > log_u:
                    t_pred.append(1)
                elif log_f > log_t and log_f > log_u:
                    t_pred.append(0)
                else:
                    t_pred.append(-1)
                conf = abs(max([log_t, log_f, log_u]) - min([log_t, log_f, log_u]))
                t_conf.append(conf)
        else:
            # If theres more than one transaction, split whole timeframe into ten buckets, make predicion up to each bucket
            first_ts = int(t[0][2])
            last_ts = int(t[-1][2])
            diff = int(round((last_ts - first_ts) * 0.1 + 0.5))
            first_ts += diff
            last_ts += diff
            for i in range(first_ts, last_ts, diff):
                this_t = np.asarray([l for l in t if int(l[2]) <= i])

                # Print the transactions that are passed to the model
                # print([list(map(int,t)) for t in this_t])
                log_t = model_t.score(this_t)
                log_f = model_f.score(this_t)
                log_u = model_u.score(this_t)

                if log_t > log_f and log_t > log_u:
                    t_pred.append(1)
                elif log_f > log_t and log_f > log_u:
                    t_pred.append(0)
                else:
                    t_pred.append(-1)
                conf = abs(max([log_t, log_f, log_u]) - min([log_t, log_f, log_u]))
                t_conf.append(conf)
        # Print prediction after each bucket and actual truth value
        # print(t_pred, truth[-1])
        confidence_all_classes.append([log_t, log_f, log_u])
        confidence.append(t_conf)
        pred.append(t_pred[-1])
        iter_pred.append(t_pred)

    match = [1 if t == p else 0 for t, p in zip(truth, pred)]

    corclassified_hash = [hsh for hsh, m in zip(facts_test.as_matrix(['hash']).flatten(), match) if m == 1]
    misclassified_hash = [hsh for hsh, m in zip(facts_test.as_matrix(['hash']).flatten(), match) if m == 0]
    misclassified_per_label = [truth[idx] for idx, m in enumerate(match) if m == 0]

    f1_score = sklearn.metrics.f1_score(truth, pred, average='micro')
    print('F1 Score: {}'.format(f1_score))
    print('Accuracy: {}, Correct: {}, Total: {}'.format(sum(match) / (len(match) * 1.0), sum(match), len(match)))

    result_analysis(facts, transactions, truth, pred, iter_pred, confidence)
    roc_analysis(truth, confidence_all_classes)
    precision_recall(truth, pred, 3)
    return f1_score, misclassified_per_label, misclassified_hash, corclassified_hash


def result_analysis(facts, transactions, truth, pred, iter_pred, confidence):
    # Details on accuracy, confidence, and labels
    match = [1 if t == p else 0 for t, p in zip(truth, pred)]
    match_per_iter = [[1 if p == truth else 0 for p in prediction] for truth, prediction in zip(truth, iter_pred)]
    match_per_iter_t = list(map(list, zip(*match_per_iter)))
    accuracy_per_iter = [sum(per) / (1.0 * len(per)) for per in match_per_iter_t]

    conf_per_iter = [int(sum(conf) / (1.0 * len(conf))) for conf in list(map(list, zip(*confidence)))]

    early_correct = []
    for idx, p in enumerate(iter_pred):
        if not match[idx]:
            continue
        last_cor = len(p)
        for i in range(len(p) - 2, 0, -1):
            last_cor = i
            if p[i] != p[i + 1]:
                break
        early_correct.append(last_cor / (len(p) * 1.0))

    print('\tAccuracy over time in 10pp steps: {}'.format(accuracy_per_iter))
    print('\tConfidence over time in 10pp steps: {}'.format(conf_per_iter))
    print('\tCorrect class could be detected after {} % of all tweets'.format(
        sum(early_correct) / (1.0 * len(early_correct))))
    print('Categories: {}'.format(Counter(truth)))
    print('Predicted Categories: {}'.format(Counter(pred)))
    print('Missed facts with labels: {}'.format(Counter([facts['true'][idx] for idx, m in enumerate(match) if m == 0])))
    print('Missed predictions with labels: {}'.format(Counter([pred[idx] for idx, m in enumerate(match) if m == 0])))

    match = [1 if t == p else 0 for t, p in zip(truth, pred)]
    misclassified_hash = [hsh for hsh, m in zip(facts.as_matrix(['hash']).flatten(), match) if m == 0]
    corclassified_hash = [hsh for hsh, m in zip(facts.as_matrix(['hash']).flatten(), match) if m == 1]

    print('$$$$ Detailed Analyiss $$$$')
    print('\tCorrectly classified')
    classification_analysis(facts, transactions, corclassified_hash)
    print('\tIncorrectly classified')
    classification_analysis(facts, transactions, misclassified_hash)


def roc_analysis(truth, confidence_all_classes):
    n_classes = 3
    lw = 2
    lb = LabelBinarizer()
    lb.fit([0, 1, -1])
    truth_bin = lb.transform(truth)
    pred_bin = np.asarray(confidence_all_classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(truth_bin[:, i], pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(truth_bin.ravel(), pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    if SHOW_PLOTS:
        plt.show()


def precision_recall(Y_test, y_score, n_classes):
    lb = LabelBinarizer()
    lb.fit([0, 1, -1])
    Y_test = lb.transform(Y_test)
    y_score = lb.transform(y_score)
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))
    if SHOW_PLOTS:
        plt.show()


def classification_analysis(facts, transactions, classified_hash):
    support, reject = [], []
    support_true, support_false, support_unknown = [], [], []
    reject_true, reject_false, reject_unknown = [], [], []
    duration = []
    certainty = []
    topics = []
    start = []

    longest_duration = max(
        [fact_transactions.iloc[-1].timestamp - fact_transactions.iloc[0].timestamp for fact_transactions in
         [transactions[transactions.fact == hsh].sort_values(by=['fact', 'timestamp']) for hsh in classified_hash]])

    for hsh in classified_hash:
        fact_transactions = transactions[transactions.fact == hsh].sort_values(by=['fact', 'timestamp'])

        topics.extend(facts[facts.hash == hsh].as_matrix(['topic']).flatten())

        if str(facts[facts.hash == hsh].as_matrix(['true']).flatten()[0]) == '1':
            support_true.append(fact_transactions[fact_transactions.stance == 'supporting'].shape[0])
            reject_true.append(fact_transactions[fact_transactions.stance == 'denying'].shape[0])
        if str(facts[facts.hash == hsh].as_matrix(['true']).flatten()[0]) == '0':
            support_false.append(fact_transactions[fact_transactions.stance == 'supporting'].shape[0])
            reject_false.append(fact_transactions[fact_transactions.stance == 'denying'].shape[0])
        if str(facts[facts.hash == hsh].as_matrix(['true']).flatten()[0]) == 'unknown':
            support_unknown.append(fact_transactions[fact_transactions.stance == 'supporting'].shape[0])
            reject_unknown.append(fact_transactions[fact_transactions.stance == 'denying'].shape[0])

        support.append(fact_transactions[fact_transactions.stance == 'supporting'].shape[0])
        reject.append(fact_transactions[fact_transactions.stance == 'denying'].shape[0])

        duration.append(fact_transactions.iloc[-1].timestamp - fact_transactions.iloc[0].timestamp)

        certainty.extend(fact_transactions.as_matrix(['weight']).flatten())

        start.append(fact_transactions.iloc[0].timestamp)

        first_time = fact_transactions.iloc[0].timestamp
        hours_in_range = int(duration[-1].days + 1) * 24
        dens_s = {k: 0 for k in
                  [first_time.replace(minute=0, second=0) + timedelta(hours=dt) for dt in range(hours_in_range + 1)]}
        dens_r = {k: 0 for k in
                  [first_time.replace(minute=0, second=0) + timedelta(hours=dt) for dt in range(hours_in_range + 1)]}

        for idx, dt in fact_transactions.iterrows():
            if dt['stance'] == 'supporting':
                dens_s[dt['timestamp'].replace(minute=0, second=0)] += 1
            elif dt['stance'] == 'denying':
                dens_r[dt['timestamp'].replace(minute=0, second=0)] += 1
        tmp = {k: 0 for k in
               [first_time.replace(minute=0, second=0) + timedelta(hours=dt) for dt in range(hours_in_range + 1)]}
        # time_multi_plot(dens_s, dens_r, tmp, title=facts[facts.hash == hsh]['text'])
    print('\t\tTopcis: {}'.format(Counter(topics)))
    print('\t\tCertainty: {}'.format(Counter(certainty)))
    print('\t\tAvg supporting tweets: {}'.format(sum(support) / (1.0 * len(support))))

    print('\t\t\tAvg support tweets for true facts: {}'.format(sum(support_true) / (1.0 * len(support_true))))
    print('\t\t\tAvg support tweets for false facts: {}'.format(sum(support_false) / (1.0 * len(support_false))))
    if len(support_unknown) > 0: print(
        '\t\t\tAvg support tweets for unknown facts: {}'.format(sum(support_unknown) / (1.0 * len(support_unknown))))

    print('\t\tAvg denying tweets: {}'.format(sum(reject) / (1.0 * len(reject))))

    print('\t\t\tAvg reject tweets for true facts: {}'.format(sum(reject_true) / (1.0 * len(reject_true))))
    print('\t\t\tAvg reject tweets for false facts: {}'.format(sum(reject_unknown) / (1.0 * len(reject_false))))
    if len(reject_unknown) > 0: print(
        '\t\t\tAvg reject tweets for unknown facts: {}'.format(sum(reject_unknown) / (1.0 * len(reject_unknown))))

    print('\t\tAvg duration: {}'.format(sum(duration, timedelta(0)) / (len(duration))))
    print('\t\tAvg start: {}'.format(sum((start[0] - d_i for d_i in start), timedelta(0)) / (len(start))))


def aggregated_analysis(facts, transactions, mismatches_hsh, corrmatches_hash):
    transactions = transactions[transactions.stance != 'appeal-for-more-information']
    transactions = transactions[transactions.stance != 'comment']
    transactions = transactions[transactions.stance != 'underspecified']

    mism = sorted([[k, v] for k, v in mismatches_hsh.items()], key=lambda s: s[1])
    corm = sorted([[k, v] for k, v in corrmatches_hash.items()], key=lambda s: s[1])
    print('\n &&&&&&&&&&&&&&&&&&&&&&\n'
          'Details Correct Predictions')
    for f in corm[:5]:
        hsh = f[0]
        print(facts[facts.hash == hsh].as_matrix(['topic', 'true', 'text']).flatten())
        print(transactions[transactions.fact == hsh].sort_values(by=['timestamp']).as_matrix(
            ['timestamp', 'stance', 'weight']))
    print('\n &&&&&&&&&&&&&&&&&&&&&&\n'
          'Details False Predictions')
    for f in mism[:5]:
        hsh = f[0]
        print(facts[facts.hash == hsh].as_matrix(['topic', 'true', 'text']).flatten())
        print(transactions[transactions.fact == hsh].sort_values(by=['timestamp']).as_matrix(
            ['timestamp', 'stance', 'weight']))


def main():
    # wisdom_crowd(*get_users())
    # wisdom_certainty(*get_users())
    avg = []
    mismatches_label = []
    mismatches_hash = []
    matches_hash = []
    for i in range(2):
        try:
            score, mism_label, mism_hash, m_hash = hmm_prediction(*get_data())
            avg.append(score)
            mismatches_label.extend(mism_label)
            mismatches_hash.extend(mism_hash)
            matches_hash.extend(m_hash)
        except Exception as e:
            print('error: {}'.format(e))
    print('Average F1 Score: {}'.format(sum(avg) / (len(avg) * 1.0)))
    print('Misclassified ones: {}'.format(Counter(mismatches_label)))
    # print('Misclassified hash: {}'.format(Counter(mismatches_hash)))
    # print('Correct classified hash: {}'.format(Counter(matches_hash)))
    facts, transactions = get_data()
    # aggregated_analysis(facts, transactions, Counter(mismatches_hash), Counter(matches_hash))


if __name__ == "__main__":
    main()
