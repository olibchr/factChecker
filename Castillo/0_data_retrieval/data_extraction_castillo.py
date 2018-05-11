import sys, json, glob, os, hashlib, pandas
sys.path.insert(0, os.path.dirname(__file__) + '../../2_objects')
from Transaction import Transaction
from Fact import Fact
from User import User
from dateutil import parser
from collections import Counter
import datetime, hashlib
import numpy as np
import csv

DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/4_RawData/Castillo/'
OUT = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/5_DataCastillo/'
# DIR = '/var/scratch/obr280/0_Thesis/3_Data/'

# Objectives:
# Build Users + Facts

def get_facts_tweet():
    tweet_to_fact = {}
    with open(DIR + 'trendid-tweetid.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            tweet_to_fact[r[1]] = r[0]
    print(len(tweet_to_fact))
    return tweet_to_fact

def get_facts_y():
    fact_to_cred = {}
    with open(DIR + 'labels-trendid-credible.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            fact_to_cred[r[0]] = r[1]
    print(len(fact_to_cred))
    return fact_to_cred

def get_facts_type():
    fact_to_type = {}
    with open(DIR + 'labels-trendid-newsworthy.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            fact_to_type[r[0]] = r[1]
    print(len(fact_to_type))
    return fact_to_type


def build_facts(fact_to_cred, fact_to_type, fact_to_user):
    facts = []
    for f in fact_to_cred.items():
        true = 0 if f[1] == 'NOTCREDIBLE' else 1
        type = fact_to_type[f[0]] if f[0] in fact_to_type else 0
        # <RUMOR_TPYE, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND, SOURCE_TWEET, ?HASH>
        this_fact = Fact(type, '', '', true, 0, 0, 0, hash=f[0])
        facts.append(this_fact)
    return facts


def store_result():
    with open(DIR + '../facts.json', 'w') as out_file:
        out_file.write(json.dumps([f.__dict__ for f in FACTS]))


fact_to_cred = get_facts_y()
fact_to_type = get_facts_type()
tweet_to_fact = get_facts_tweet()
tweet_to_fact = {k:v for k,v in tweet_to_fact.items() if v in fact_to_cred}
print(len(tweet_to_fact))
FACTS = build_facts(fact_to_cred, fact_to_type, tweet_to_fact)
store_result()