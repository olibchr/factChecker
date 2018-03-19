from datetime import datetime
import warnings, json, glob, subprocess, time
import sys, os
from six.moves import urllib

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import requests
import re, nltk
from User import User
from Transaction import Transaction
from IPython.display import HTML
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/'
SUBSCRIPTION_KEY = "28072856698a426799cbab6f002c741b"


def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


def user_decoder(obj):
    if 'user_id' not in obj.keys(): return obj
    # <user_id, tweets, fact, transactions, credibility, controversy>
    return User(obj['user_id'], obj['tweets'], obj['fact'], obj['transactions'], obj['credibility'],
                obj['controversy'])


def get_data():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=user_decoder)
        print(user.user_id)
        yield user


def parse_link_in_tweet(tweet):
    pass

def get_web_doc_sentences(url):
    print(url)
    try:
        html_document = urllib.request.urlopen(url)
        soup = BeautifulSoup(html_document.read(), 'lxml')
        for script in soup(["script", "style"]):
            script.decompose()
        # get text
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        sentences = nltk.sent_tokenize(text)
        for idx, sen in enumerate(sentences):
            sentences[idx] = sen.replace('\n', ' ')
        return sentences
    except Exception as e:
        print('Error: {}, for url: {}'.format(e, url))
        return None


def get_bing_documents_for_tweet(user):
    search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
    print('Web search for: {}'.format(user.user_id))
    tweets_with_docs = []
    for tweet in user.tweets:
        # If tweet contains link to another tweet, link should be resolved and tweet parsed..
        parse_link_in_tweet(tweet['text'])

        search_term = tweet['text']
        print(search_term)
        headers = {"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY}
        params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "answerCount": 30, "promote": "News"}
        # params  = {"q": search_term, "textDecorations":True, "textFormat":"HTML", "count": 30, "promote": "News"}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        web_documents = {'tweet': tweet['text'],
                         'created_at': tweet['created_at'],
                         'docs': [
                             {
                                 'url': doc['url'],
                                 'content': get_web_doc_sentences(doc['url'])
                             } for doc in search_results['webPages']['value']]
                         }
        tweets_with_docs.append(web_documents)
    return tweets_with_docs


def calculate_similarity(tweet, web_documents):
    ungrams = nltk.sent_tokenize(tweet)
    possible_snippets = [''.join(web_documents[i:i + 4]) for i in range(len(web_documents) - 4)]

    pass


def store_result(user):
    with open(DIR + 'user_tweet_web_search/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')


def main():
    users = get_data()
    for user in users:
        user.tweets = get_bing_documents_for_tweet(user)
        store_result(user)


if __name__ == "__main__":
    main()
