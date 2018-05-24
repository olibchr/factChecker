from datetime import datetime
import warnings, json, glob, subprocess, time
import sys, os

sys.path.insert(0, os.path.dirname(__file__) + '../../2_objects')
from Fact import Fact
from User import User
from Transaction import Transaction
from decoder import decoder
import tweepy
from collections import Counter

ALT_ACCOUNT = False

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

DIR = os.path.dirname(__file__) + '../../../5_DataCastillo/'


def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


def get_data():
    fact_file = glob.glob(DIR + 'facts.json')[0]
    transactions_file = glob.glob(DIR + 'factTransaction.json')[0]
    facts = json.load(open(fact_file), object_hook=decoder)
    transactions = json.load(open(transactions_file), object_hook=decoder)
    if ALT_ACCOUNT:
        transactions = sorted(transactions, reverse=True, key=lambda t: t.user_id)
    else:
        transactions = sorted(transactions, reverse=False, key=lambda t: t.user_id)
    user_files = [user_file for user_file in glob.glob(DIR + 'user_tweets/' + 'user_*.json') if
                  'user_' in user_file]
    user_files = [user_file[user_file.rfind('_') + 1:user_file.rfind('.')] for user_file in user_files]
    print(len(user_files))
    return facts, transactions, user_files


def get_user_tweets(api, transactions, user_files):
    i = 0
    print(len(transactions))
    while i < len(transactions):
        tr = transactions[i]
        i += 1
        user_id = tr.user_id
        if str(user_id) in user_files:
            print("User {} has been crawled before".format(user_id))
            continue
        try:
            user_tweets = []
            user_features = None
            for status in tweepy.Cursor(api.user_timeline, id=user_id).items():
                parsed_status = {'text': status._json['text'],
                                 'created_at': status._json['created_at'],
                                 'reply_to': status._json['in_reply_to_status_id'],
                                 'retweets': status._json['retweet_count'],
                                 'favorites': status._json['favorite_count']}
                if not user_features:
                    user_features = {
                        'name': status._json['user']['name'],
                        'location': status._json['user']['location'],
                        'followers': status._json['user']['followers_count'],
                        'friends': status._json['user']['friends_count'],
                        'description': status._json['user']['description'],
                        'created_at': status._json['user']['created_at'],
                        'verified': status._json['user']['verified'],
                        'statuses_count': status._json['user']['statuses_count'],
                        'lang': status._json['user']['lang']
                    }
                if 'quoted_status' in status._json:
                    parsed_status['quoted_status'] = {
                        'created_at': status._json['quoted_status']['created_at'],
                        'text': status._json['quoted_status']['text']
                    }
                user_tweets.append(parsed_status)
            # <user_id, tweets, fact, transactions, credibility, controversy>
            this_user = User(user_id, transactions=tr, tweets=user_tweets, features=user_features)
            print('Got tweets for user: {}, found: {}'.format(user_id, len(this_user.tweets)))
            user_files.append(str(user_id))
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


def get_users():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    print('{} users'.format(len(user_files)))
    if len(user_files) < 10: print('WRONG DIR?')
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=decoder)
        yield user


def was_user_correct(user, facts, transactions):
    for tr in transactions:
        if str(tr.user_id) == str(user.user_id):
            transaction = tr
            user.transactions = tr
            transactions.remove(tr)
            break
    for fact in facts:
        if fact.hash == transaction.fact:
            user.text = transaction.text
            if (str(fact.true) == '1' and transaction.stance == 'supporting') or (
                            str(fact.true) == '0' and transaction.stance == 'denying'):
                user.was_correct = 1
            elif(str(fact.true) == '1' and transaction.stance == 'denying') or \
                    (str(fact.true) == '0' and transaction.stance == 'supporting'):
                user.was_correct = 0
            else:
                user.was_correct = -1
            print(fact.true, transaction.stance, user.was_correct)
    return user


def store_result(user):
    # TODO: was previosuly append, need to scan files and make sure we dont have duplicates
    # Todo: can just check if files have more than one line and omit other lines
    with open(DIR + 'user_tweets/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')


def main():
    print(DIR)
    if ALT_ACCOUNT:
        print("Using alt account")
    else:
        print("Using reg account")
    facts, transactions, user_files = get_data()
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)
    users = get_user_tweets(api, transactions, user_files)
    #users = get_users()
    users = [was_user_correct(user, facts, transactions) for user in users]
    # print(Counter([u.was_correct for u in users]))
    for user in users:
        store_result(user)


if __name__ == "__main__":
    main()
