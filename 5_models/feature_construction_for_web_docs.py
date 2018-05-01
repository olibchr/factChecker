import glob, os, sys, json, datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import re, nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from Fact import Fact
from User import User
from Transaction import Transaction
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

SERVER_RUN = True

DIR = os.path.dirname(__file__) + '../../3_Data/'


def decoder(o):
    def user_decoder(obj):
        if 'user_id' not in obj.keys(): return obj
        # if 'avg_time_to_retweet' in obj.keys():
        return User(obj['user_id'], tweets=obj['tweets'], fact=obj['fact'], transactions=obj['transactions'],
                    credibility=obj['credibility'],
                    controversy=obj['controversy'], features=obj['features'], was_correct=obj['was_correct'],
                    avg_time_to_retweet=obj['avg_time_to_retweet'] if 'avg_time_to_retweet' in obj.keys() else None,
                    sent_tweets_density=obj['sent_tweets_density'] if 'sent_tweets_density' in obj.keys() else None,
                    sent_tweets_avg=obj['sent_tweets_avg'] if 'sent_tweets_avg' in obj.keys() else None
                    )

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
    if isinstance(o, type(datetime)):
        return o.__str__()


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

def dirty_json_parse(f):
    _decoder = json.JSONDecoder()
    def loads(s):
        """A generator reading a sequence of JSON values from a string."""
        while s:
            s = s.strip()
            obj, pos = _decoder.raw_decode(s)
            if not pos:
                raise ValueError('no JSON object found at %i' % pos)
            yield obj
            s = s[pos:]
    wrapped = list(loads(f))
    result = []
    for l in wrapped:
        for obj in l:
            result.append(obj)

def get_web_doc(user):
    doc_dir = DIR + 'user_docs_test/' + '*.json'
    doc_file = [f for f in glob.glob(doc_dir) if str(user.user_id) in f]
    if len(doc_file) == 0: return None
    doc_file = doc_file[0]
    with open(doc_file, 'r') as f:
        data = f.read()
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    ansi_escape.sub('', data)
    data = dirty_json_parse(data)
    web_docs_df = pd.DataFrame(data)
    web_docs_df.drop_duplicates(subset=['link'], inplace=True)
    print(web_docs_df.shape)
    return web_docs_df


def feature_user_web_doc_sentiment(users):
    sid = SentimentIntensityAnalyzer()
    err_list = []
    for user in users:
        web_docs_df = get_web_doc(user)
        #except Exception as e:
        #    print(e)
        #    print('%%%%: {}'.format(user.user_id))
        #    err_list.append(user.user_id)

        #if web_docs_df is None: print(user.user_id); continue
        #web_docs_df['sentiment'] = web_docs_df['content'].map(lambda x: sid.polarity_scores(x)['compound'])
        #web_docs_df.drop('content')
    print(err_list)



# <user_id, tweets, fact, transactions, credibility, controversy, features, was_correct, snippets, avg_time_to_retweet>
# tweets <text, created_at, reply_to, retweets, favorites, *quoted_status<created_at, text>>
def main():
    feature_user_web_doc_sentiment(get_users())


if __name__ == "__main__":
    main()
