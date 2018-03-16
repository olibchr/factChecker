from datetime import datetime
import warnings, json, glob
import sys, os, requests
from SPARQLWrapper import SPARQLWrapper, JSON

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
from Fact import Fact
from Transaction import Transaction

DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/'


def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


def fact_decoder(obj):
    # <RUMOR_TPYE, HASH, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND, SOURCE_TWEET>
    return Fact(obj['rumor_type'], obj['topic'], obj['text'], obj['true'], obj['proven_false'],
                obj['is_turnaround'], obj['source_tweet'], hash=obj['hash'])

def transaction_decoder(obj):
    # <sourceID, id, user_id, fact, timestamp, stance, weight>
    return Transaction(obj['sourceID'], obj['id'], obj['user_id'], obj['fact'], obj['timestamp'], obj['stance'],
                       obj['weight'])


def get_data():
    facts = json.load(open(glob.glob(DIR + 'facts.json')[0]), object_hook=fact_decoder)
    transactions = json.load(open(glob.glob(DIR + 'factTransaction.json')[0]), object_hook=transaction_decoder)
    return facts, transactions

def get_user_tweets(userID):
    


def store_result(facts):
    with open(DIR + 'user_tweets/' + 'factTransactions_annotated.json', 'w') as out_file:
        out_file.write(json.dumps([f.__dict__ for f in facts], default=datetime_converter))


def main():
    facts, transactions = get_data()
    store_result(facts)


if __name__ == "__main__":
    main()
