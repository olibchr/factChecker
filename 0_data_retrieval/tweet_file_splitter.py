import glob
from datetime import datetime
import warnings, json, glob, subprocess, time
import sys, os
from six.moves import urllib
sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import requests
import re, nltk
from User import User

DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/'

def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


def user_decoder(obj):
    if 'user_id' not in obj.keys(): return obj
    # <user_id, tweets, fact, transactions, credibility, controversy>
    return User(obj['user_id'], obj['tweets'], obj['fact'], obj['transactions'], obj['credibility'],
                obj['controversy'])


def get_user():
    user_file = glob.glob(DIR + 'user_tweets/' + 'users.json')[0]
    with open(user_file, 'r') as userfile:
        lines = userfile.readlines()
        for line in lines:
            yield json.loads(line, object_hook=user_decoder)

users = get_user()

for user in users:
    with open(DIR + 'user_tweets/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')
