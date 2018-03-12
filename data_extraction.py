import sys, json, glob, os, hashlib, pandas
from Transaction import Transaction
from Fact import Fact
from dateutil import parser
from collections import Counter
import datetime, hashlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/RAW/'
LANG = 'both'
THREADS = 'threads/' + LANG + '/'
ANNOTATIONS = 'annotations/'

A_FILE = '[' + ','.join([a for a in open(glob.glob(DIR + ANNOTATIONS + 'en' + '*.json')[0]).read().split('\n') if len(a) > 0 and a[0] != '#'])\
         + ',' + ','.join([a for a in open(glob.glob(DIR + ANNOTATIONS + 'de' + '*.json')[0]).read().split('\n') if len(a) > 0 and a[0] != '#']) + ']'
ANNOTATIONS_EN = json.loads(A_FILE)
FACTS = []
FACTS_DUMP = []
TRANSACTIONS_DUMP = []
TRANSACTIONS = []
FACT_MEMORY = []

def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

def get_annot_by_tweet_id(tweet_id):
    for annot in ANNOTATIONS_EN:
        if annot['tweetid'] == str(tweet_id):
            return annot

def time_multi_plot(d1, d2, d3, title='All tweets per day'):
    # Plot a dictionary with days as keys and count as values
    id_true = sorted([[k,d1[k], d2[k], d3[k]] for k in d1.keys()])
    df = pandas.DataFrame(id_true, columns=['date', 'support', 'deny', 'comment'])
    # df = df[['date', 'support','deny']]
    df.set_index('date',inplace=True)
    fig, ax = plt.subplots(figsize=(15,7))
    df.plot(ax=ax)
    #set ticks every week
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid()
    #
    plt.title(title)
    plt.show()


def time_plot(data, title='All tweets per day'):
    # Plot a dictionary with days as keys and count as values
    id_true = sorted([[k,v] for k,v in data.items()])
    df = pandas.DataFrame(id_true, columns=['date', 'count'])
    df.set_index('date',inplace=True)
    fig, ax = plt.subplots(figsize=(15,7))
    df.plot(ax=ax)
    #set ticks every week
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid()
    #
    plt.title(title)
    plt.show()


def stats():
    print('Number of facts: {}'.format(len(FACTS_DUMP)))
    print('Number of News categories: {}'.format(len(set([f[2] for f in FACTS_DUMP]))))
    print('Fact Veracity: {}'. format(Counter(f[4] for f in FACTS_DUMP)))
    print('Example: Type: {}, Category: {}, Topic: {}, True: {}, Was Proven False: {}, Is Turnaround: {}'.format(FACTS_DUMP[10][0], FACTS_DUMP[10][2], FACTS_DUMP[10][3], FACTS_DUMP[10][4], FACTS_DUMP[10][5], FACTS_DUMP[10][6]))
    print('\n----------------------------------\n')
    print('Number of Transactions: {}'.format(len((TRANSACTIONS_DUMP))))
    print('Number of Users: {}'.format(len(set([t[1] for t in TRANSACTIONS_DUMP]))))
    print('Stance Distribution: {}'.format(Counter([t[5] for t in TRANSACTIONS_DUMP])))
    print('Certainty Distribution: {}'.format(Counter([t[6] for t in TRANSACTIONS_DUMP])))
    print('Example: ID: {}; User ID: {}; Fact ID: {}, Timestamp: {}; Stance: {}; Certainty: {}'.format(TRANSACTIONS_DUMP[10][1], TRANSACTIONS_DUMP[10][2], TRANSACTIONS_DUMP[10][3], TRANSACTIONS_DUMP[10][4], TRANSACTIONS_DUMP[10][5], TRANSACTIONS_DUMP[10][6]))
    # Plots
    # All Tweets per day
    dts = sorted([t[4] for t in TRANSACTIONS_DUMP])
    perday_true = {k:0 for k in [dts[0].replace(hour=0, minute=0, second=0) + datetime.timedelta(days=dt) for dt in range((dts[-1]-dts[0]).days + 1)]}
    for dt in dts:
        perday_true[dt.replace(hour=0, minute=0, second=0)] += 1

    #
    time_plot(perday_true)
    # All tweets per topic
    for t in set([f[2] for f in FACTS_DUMP]):
        all_stories = [f[1] for f in FACTS_DUMP if f[2] == t]
        dts = sorted([t[4] for t in TRANSACTIONS_DUMP if t[3] in all_stories])
        #print(all_stories, len(dts))
        hours_in_range = int((max(dts)-min(dts)).days + 1) *24
        perday_true = {k:0 for k in [dts[0].replace(minute=0, second=0) + datetime.timedelta(hours=dt) for dt in range(hours_in_range + 1)]}
        perday_false = {k:0 for k in [dts[0].replace(minute=0, second=0) + datetime.timedelta(hours=dt) for dt in range(hours_in_range + 1)]}
        perday_comment = {k:0 for k in [dts[0].replace(minute=0, second=0) + datetime.timedelta(hours=dt) for dt in range(hours_in_range + 1)]}
        for dt in sorted([t for t in TRANSACTIONS_DUMP if t[3] in all_stories], key=lambda t:t[4]):
            if dt[5] == 'supporting':
                perday_true[dt[4].replace(minute=0, second=0)] += 1
            elif dt[5] == 'denying':
                perday_false[dt[4].replace(minute=0, second=0)] += 1
            else:
                perday_comment[dt[4].replace(minute=0, second=0)] += 1
        time_multi_plot(perday_true, perday_false, perday_comment, title=t)


def data_extraction():
    for TOPIC in sorted([f for f in os.listdir(DIR + THREADS) if not f.startswith('.')]):
        for THREAD in sorted([f for f in os.listdir(DIR + THREADS + TOPIC) if not f.startswith('.')]):
            THIS_THREAD = DIR + THREADS + TOPIC + '/' + THREAD + '/'
            # load thread specific jsons
            tweet_annot = json.load(open(glob.glob(THIS_THREAD + 'annotation.json')[0]))
            source_tweet = json.load(open(glob.glob(THIS_THREAD + 'source-tweets/' + '*.json')[0]))
            reaction_files = glob.glob(THIS_THREAD + 'reactions/' + '*.json')
            reactions = json.loads('[' + ','.join([(open(f).read()) for f in reaction_files]) + ']')
            # Fact
            # <RUMOR_TYPE, HASH, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND>
            truth_value = tweet_annot['true'] if 'true' in tweet_annot else 'unknown'
            is_turnaround = tweet_annot['is_turnaround'] if 'is_turnaround' in tweet_annot else 'unknown'
            if hashlib.md5(tweet_annot['category'].encode()).hexdigest() not in FACT_MEMORY:
                fact = Fact(tweet_annot['is_rumour'], TOPIC, tweet_annot['category'], truth_value, tweet_annot['misinformation'], is_turnaround)
                FACTS.append(fact)
                FACTS_DUMP.append([fact.rumor_type, fact.hash, fact.topic, fact.text, fact.true, fact.proven_false, fact.is_turnaround])
                FACT_MEMORY.append(hashlib.md5(tweet_annot['category'].encode()).hexdigest())
            # Transaction
            # <ID, USER ID, FACT ID, TIMESTAMP, {support, reject}, weight>
            source_annot = get_annot_by_tweet_id(source_tweet['id'])
            initiator = [source_tweet['id'], source_tweet['id'], source_tweet['user']['id'], fact.hash, parser.parse(source_tweet['created_at']), source_annot['support'], source_annot['certainty']]
            # dump first transaction on topic
            TRANSACTIONS_DUMP.append(initiator)
            TRANSACTIONS.append(Transaction(initiator[0], initiator[1], initiator[2], initiator[3], initiator[4], initiator[5], initiator[6]))
            # parse and dump all other transactions
            for reaction in reactions:
                annot = get_annot_by_tweet_id(reaction['id'])
                if annot == None:
                    #print('ANNOTATION NOT FOUND: {}'.format(reaction['id']))
                    continue
                if 'responsetype-vs-source' not in annot:
                    #print('STANCE NOT FOUND: {}'.format(reaction['id']))
                    continue
                #
                # Map Stance
                stance = annot['responsetype-vs-source']
                if source_annot['support'] == 'supporting':
                    if annot['responsetype-vs-source'] == 'agreed':
                        stance = 'supporting'
                    if annot['responsetype-vs-source'] == 'disagreed':
                        stance = 'denying'
                elif source_annot['support'] == 'denying':
                    if annot['responsetype-vs-source'] == 'agreed':
                        stance = 'denying'
                    if annot['responsetype-vs-source'] == 'disagreed':
                        stance = 'supporting'
                elif source_annot['support'] == 'underspecified':
                    if annot['responsetype-vs-source'] == 'agreed':
                        stance = 'denying'
                    if annot['responsetype-vs-source'] == 'disagreed':
                        stance = 'supporting'
                elif annot['responsetype-vs-source'] == 'appeal-for-more-information':
                    stance = 'underspecified'
                else:
                    if annot['responsetype-vs-source'] != 'comment': print(source_annot['support'], annot['responsetype-vs-source'])
                #
                #
                certainty = annot['certainty'] if 'certainty' in annot else 'underspecified'
                #
                comment = [source_tweet['id'], reaction['id'], reaction['user']['id'], fact.hash, parser.parse(reaction['created_at']), stance, certainty]
                TRANSACTIONS_DUMP.append(comment)
                TRANSACTIONS.append(Transaction(comment[0], comment[1], comment[2], comment[3], comment[4], comment[5], comment[6]))

def store_result():
    with open(DIR + 'factTransaction.json', 'w') as out_file:
        out_file.write(json.dumps([t.__dict__ for t in TRANSACTIONS], default=datetime_converter))

    with open(DIR + 'facts.json', 'w') as out_file:
        out_file.write(json.dumps([f.__dict__ for f in FACTS], default=datetime_converter))


def main():
    data_extraction()
    store_result()
    #stats()


if __name__ == "__main__":
    main()

