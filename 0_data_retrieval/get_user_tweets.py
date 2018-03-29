from datetime import datetime
import warnings, json, glob, subprocess, time
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
from Fact import Fact
from User import User
from Transaction import Transaction
import tweepy

SERVER_RUN = True
ALT_ACCOUNT = True

# Generate your own at https://apps.twitter.com/app
CONSUMER_KEY = '4Y897wHsZJ2Qud1EgncojnoNS'
CONSUMER_SECRET = 'sMpckIKpf00c1slGciCe4FvWlUTkFUGKkMAu88x2SBdJRW3laR'
OAUTH_TOKEN = '1207416314-pX3roPjOm0xNuGJxxRFfE6H0CyHRCgnzXvNfFII'
OAUTH_TOKEN_SECRET = 'NVS29lZafbCF4kvc1yCEKg0f00AYE3Ogj7XkygHsBI5LD'
if ALT_ACCOUNT:
    CONSUMER_KEY = '0pUhFi92XQbPTB70eEnhJ0fTH'
    CONSUMER_SECRET = 'DLjLTOoonzO5ADVfIppnLmMpCL1qM9fOHkhoXfXYIQXe3hvC9W'
    OAUTH_TOKEN = '978935525464858624-uBlhj4nIUr2eEJghiNkSzFO25hcHW2I'
    OAUTH_TOKEN_SECRET = 'eqgP2jzCzJVqcWxaqwTbFeHWKjDvMEKD6YR78UNhse6qp'


DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/'
if SERVER_RUN: DIR = '/var/scratch/obr280/0_Thesis/3_Data/'


def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


def fact_decoder(obj):
    # <RUMOR_TPYE, HASH, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND, SOURCE_TWEET>
    return Fact(obj['rumor_type'], obj['topic'], obj['text'], obj['true'], obj['proven_false'],
                obj['is_turnaround'], obj['source_tweet'], hash=obj['hash'])


def transaction_decoder(obj):
    # <sourceId, id, user_id, fact, timestamp, stance, weight>
    return Transaction(obj['sourceId'], obj['id'], obj['user_id'], obj['fact'], obj['timestamp'], obj['stance'],
                       obj['weight'])


def get_data():
    fact_file = glob.glob(DIR + 'facts.json')[0]
    transactions_file = glob.glob(DIR + 'factTransaction.json')[0]
    facts = json.load(open(fact_file), object_hook=fact_decoder)
    transactions = json.load(open(transactions_file), object_hook=transaction_decoder)
    if ALT_ACCOUNT: transactions = sorted(transactions, reverse=True, key=lambda t: t.user_id)
    else: transactions = sorted(transactions, reverse=False, key=lambda t: t.user_id)
    user_files = [user_file for user_file in glob.glob(DIR + 'user_tweets/' + 'user_*.json') if 'user_' in user_file]
    user_files = [user_file[user_file.rfind('_')+1:user_file.rfind('.')] for user_file in user_files]
    return facts, transactions, user_files


def get_user_tweets(api, transactions, user_files):
    i = 0
    while i < len(transactions):
        tr = transactions[i]
        i += 1
        user_id = tr.user_id
        if str(user_id) in user_files:
            print("User {} has been crawled before".format(user_id))
            continue
        try:
            user_tweets = []
            for status in tweepy.Cursor(api.user_timeline, id=user_id).items():
                parsed_status = {'text':status._json['text'], 'created_at':status._json['created_at']}
                if 'quoted_status' in status._json:
                    parsed_status['quoted_status'] = status._json['quoted_status']['text']
                user_tweets.append(parsed_status)
            # <user_id, tweets, fact, transactions, credibility, controversy>
            this_user = User(user_id, transactions=tr, tweets=user_tweets)
            user_files.append(str(user_id))
            print('Got tweets for user: {}, found: {}'.format(user_id, len(this_user.tweets)))
            yield this_user
        except tweepy.error.TweepError as e:
            print('Twitter error: {}'.format(e))
            if 'status code = 429' in str(e):
                print('Going to sleep for 15 min..  zzzz')
                time.sleep(900)
                i -= 1
        except Exception as e:
            print('Error: {}'.format(e))
            yield User(tr.user_id, transactions=tr)


def store_result(user):
    with open(DIR + 'user_tweets/' + 'user_' + str(user.user_id) + '.json', 'a') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')


def main():
    print(DIR)
    if ALT_ACCOUNT: print("Using alt account")
    else: print("Using reg account")
    facts, transactions, user_files = get_data()
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)
    users = get_user_tweets(api, transactions, user_files)
    for user in users:
        store_result(user)

    #store_result(users.values())

if __name__ == "__main__":
    main()
