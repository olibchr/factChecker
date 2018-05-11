import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from User import User
from Fact import Fact
from Transaction import Transaction

def decoder(o):
    def user_decoder(obj):
        if 'user_id' not in obj.keys(): return obj
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
                           obj['weight'], obj['text'])

    if 'tweets' in o.keys():
        return user_decoder(o)
    elif 'hash' in o.keys():
        return fact_decoder(o)
    elif 'sourceId' in o.keys():
        return transaction_decoder(o)
    else:
        return o