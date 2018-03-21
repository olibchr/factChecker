from datetime import datetime
import warnings, json, glob, subprocess, time
import sys, os, multiprocessing
from six.moves import urllib

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import requests
import re, nltk
from User import User
from Transaction import Transaction
from IPython.display import HTML
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()
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


def skip_pre_searched_tweets(user):
    with open(DIR + 'user_tweet_web_search/' + 'user_' + str(user.user_id) + '.json', 'r') as in_file:
        pre_crawled_docs = [json.loads(line) for line in in_file.readlines()]
        pre_crawled_docs = [d['text'] for d in pre_crawled_docs if 'text' in d]
    tweets_to_be_searched = []
    for t in user.tweets:
        if t['text'] not in pre_crawled_docs:
            tweets_to_be_searched.append(t)
    user.tweets = tweets_to_be_searched
    return user


def get_data():
    user_files = glob.glob(DIR + 'user_tweets/' + 'user_*.json')
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=user_decoder)
        yield user


def parse_links_in_tweet(tweet):
    tweet_text = tweet['text']
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet_text)
    for url in urls:
        if 'https://t.co/' in url:
            tweet_text = tweet_text.replace(url, '')
    if 'quoted_status' in tweet:
        tweet_text = tweet_text + ' ' + tweet['quoted_status']
    return tweet_text


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
    # Create user file if it does not exist
    if user.user_id not in pre_crawled_files:
        create_user_file(user)
    for tweet in user.tweets:
        # If tweet contains link to another tweet, link should be resolved and tweet parsed..
        search_term = parse_links_in_tweet(tweet)
        print(search_term)

        headers = {"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY}
        params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "answerCount": 1, "promote": "News", "mkt": "en-US"}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        relevant_pages = search_results['webPages']['value'] if 'webPages' in search_results else None
        # user.tweets <text, created_at, quoted_status, docs>
        docs_formatted = [{
                              'url': doc['url'],
                              'content': get_web_doc_sentences(doc['url'])
                          } for doc in relevant_pages] if relevant_pages is not None else []
        tweet['docs'] = docs_formatted
        # It would be easier to find unigrams and bigrams right away instead of storing the whole document
        for doc in docs_formatted:
            if doc['content']: get_ngram_snippets(tweet, doc['content'])
        # Append search result to user file
        store_search_result(user.user_id, tweet)


def get_ngram_snippets(tweet, web_document):
    # web doc in sentences
    # Todo: lemmatization on tweet & doc to find better unigrames / bigrams?
    uni_tweet_tokens = nltk.word_tokenize(tweet['text'])
    bi_tweet_tokens = [' '.join(uni_tweet_tokens[i:i + 2]) for i in range(len(uni_tweet_tokens) - 1)]

    if len(web_document) < 4:
        doc_snippets = ' '.join(web_document)
    else:
        doc_snippets = [' '.join(web_document[i:i + 4]) for i in range(len(web_document) - 3)]
    print(doc_snippets)
    OVERLAP_THRESHOLD = 0.4

    unigram_snippets = []
    bigram_snippets = []
    for snip in doc_snippets:
        cnt_u = 0
        cnt_b = 0
        for unigr in uni_tweet_tokens:
            if unigr in snip: cnt_u += 1
        overlap_score_u = cnt_u / len(uni_tweet_tokens)
        if overlap_score_u >= OVERLAP_THRESHOLD:
            print("Uni Overlap: {}".format(snip))
            unigram_snippets.append([snip, overlap_score_u])
        for bigr in bi_tweet_tokens:
            if bigr in snip: cnt_b += 1
        overlap_score_b = cnt_b / len(bi_tweet_tokens)
        if overlap_score_b >= OVERLAP_THRESHOLD:
            print("Bi Overlap: {}".format(snip))
            bigram_snippets.append([snip, overlap_score_b])

    # Unigrams<Snippets<text, score>>, Bigrams<Snippets<text, score>>
    print(len(unigram_snippets))
    print(len(bigram_snippets))
    return unigram_snippets, bigram_snippets


def create_user_file(user):
    with open(DIR + 'user_tweet_web_search/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')


def store_search_result(user_id, tweet):
    with open(DIR + 'user_tweet_web_search/' + 'user_' + str(user_id) + '.json', 'a') as out_file:
        out_file.write(json.dumps(tweet, default=datetime_converter) + '\n')


def main():
    global pre_crawled_files
    pre_crawled_files = [user_file for user_file in glob.glob(DIR + 'user_tweet_web_search/' + 'user_*.json') if
                         'user_' in user_file]
    pre_crawled_files = [user_file[user_file.rfind('_') + 1:user_file.rfind('.')] for user_file in pre_crawled_files]
    users = get_data()
    # Can we parallelize this pls?
    # Parallel(n_jobs=NUM_CORES)(delayed(get_bing_documents_for_tweet)(user) for user in users)
    for user in users:
        if str(user.user_id) in pre_crawled_files:
            # Get user file, skip tweets that have been searched for already
            user = skip_pre_searched_tweets(user)
        get_bing_documents_for_tweet(user)


if __name__ == "__main__":
    main()
