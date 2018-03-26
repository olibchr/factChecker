from datetime import datetime
import warnings, json, glob, subprocess, time
import sys, os, multiprocessing
from six.moves import urllib

sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
import requests
import re, nltk, threading
from User import User
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from GoogleScraper import scrape_with_config, GoogleSearchError

NUM_CORES = 8 # multiprocessing.cpu_count()
DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/'
DIR = '/var/scratch/obr280/0_Thesis/3_Data/'
SUBSCRIPTION_KEY = "28072856698a426799cbab6f002c741b"

WNL = WordNetLemmatizer()
NLTK_STOPWORDS = set(stopwords.words('english'))
OVERLAP_THRESHOLD = 0.4


def make_web_query(keywords, userid):
    config = {
        'use_own_ip': 'True',
        'keywords': keywords,
        'search_engines': ['bing'],
        'num_pages_for_keyword': 1,
        'scrape_method': 'http',
        'do_caching': 'True',
        'output_filename': '../99_tmp/' + str(userid) + '_search_results.csv'
    }
    try:
        search = scrape_with_config(config)
    except GoogleSearchError as e:
        print(e)
        return None
    if not search.serps: return None

    results = []
    for idx, serp in enumerate(search.serps):
        links = []
        for link in serp.links:
            links.append(link)
        results.append({'query': keywords[idx], 'results': links, 'serp': str(serp)})
    return results


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
    if len(user_files) < 10: print('WRONG DIR?')
    for user_file in user_files:
        user = json.loads(open(user_file).readline(), object_hook=user_decoder)
        yield user


def extract_query_term(tweet):
    tweet_text = tweet['text']
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet_text)
    for url in urls:
        if 'https://t.co/' in url:
            tweet_text = tweet_text.replace(url, '')
    if 'quoted_status' in tweet:
        tweet_text = tweet_text + ' ' + tweet['quoted_status']
    if tweet_text[:2].lower() == 'rt': tweet_text = tweet_text[2:]
    return tweet_text


def get_web_doc(url):
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
        text = '\n'.join(chunk for chunk in chunks if chunk).lower()

        return text
    except Exception as e:
        print('Parsing error: {}, for url: {}'.format(e, url))
        return None


def is_tweet_fact(tweet):
    # Heuristics to consider a tweet an expression towards some factual statement
    # If the length of the tweet is at least 5 words
    # If the tweet is a response to another tweet
    # If the tweet contains an url
    # If the tweet contains NE's
    tweet_text = tweet['text'].replace('@', '').replace('#', '')
    tokenized = nltk.word_tokenize(tweet_text)

    if len(tokenized) >= 5:
        return True
    if 'quoted_status' in tweet:
        return True

    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet['text'])
    if urls: return True

    tagged = nltk.pos_tag(tokenized)
    parse_tree = nltk.ne_chunk(tagged, binary=True)
    for t in parse_tree.subtrees():
        if t.label() == 'NE':
            return True
    return False


def get_bing_documents_for_tweet(user):
    print('Web search for: {}'.format(user.user_id))
    # Create user file if it does not exist
    if user.user_id not in pre_crawled_files:
        create_user_file(user)
    search_terms = []
    tweets = user.tweets
    for tweet in tweets:
        # user.tweets <text, created_at, quoted_status>
        # if not is_tweet_fact(tweet): continue
        # If tweet contains link to another tweet, link should be resolved and tweet parsed..
        if not is_tweet_fact(tweet):
            # print('Gonna skip: {}'.format(tweet['text']))
            continue
        query_term = extract_query_term(tweet)
        tweet['search_instance'] = len(search_terms)
        if is_tweet_fact(tweet):
            search_terms.append(query_term)

    print(len(search_terms))
    results = make_web_query(search_terms, user.user_id)

    for tweet in tweets:
        tweet['snippets'] = []
        if tweet['search_instance']:
            search_instance = tweet['search_instance']
            del tweet['search_instance']
            query_term = results[search_instance]['query']
            relevant_pages = [result['link'] for result in results[search_instance]]

            if relevant_pages: print('Found results: {}'.format(len(relevant_pages)))
            docs_formatted = [{
                                  'url': url,
                                  'content': get_web_doc(url)
                              } for url in relevant_pages] if len(relevant_pages) > 0 else []
            # Parse unigram and bigram overlaps of search results with query
            # user.tweets <text, created_at, quoted_status, snippets<unigrams, bigrams, url>>
            for doc in docs_formatted:
                if doc['content']:
                    tweet['snippets'].append(get_ngram_snippets(query_term, doc['content'], doc['url']))
        # Append search result to user file
        store_search_result(user.user_id, tweet)


def get_ngram_snippets(tweet_text, web_document, url):
    # Remove odd characters from tweet, to lowercast and remove stopword. Parse into uni and bigrams
    tweet_text = re.sub(r'[^a-z0-9]', ' ', tweet_text.lower())

    # Todo: decide on lemmatization on tweet & doc to find better unigrames / bigrams?
    # uni_tweet_tokens = [t.strip() for t in nltk.word_tokenize(tweet_text) if t not in NLTK_STOPWORDS]
    uni_tweet_tokens = [WNL.lemmatize(i) for i in nltk.word_tokenize(tweet_text) if i not in NLTK_STOPWORDS]
    bi_tweet_tokens = [' '.join(uni_tweet_tokens[i:i + 2]) for i in range(len(uni_tweet_tokens) - 1)]
    if len(uni_tweet_tokens) == 0:
        return {'unigrams': [],
                'bigrams': [],
                'url': url
                }
    if len(bi_tweet_tokens) == 0: bi_tweet_tokens = uni_tweet_tokens

    # Split doc into senctences, remove stopwords and to lowercase
    doc_sents = nltk.sent_tokenize(web_document)
    doc_sents = [" ".join([WNL.lemmatize(i) for i in sent.split() if i not in NLTK_STOPWORDS])
                     .replace('\n', ' ')
                 for sent in doc_sents]

    if len(doc_sents) < 4:
        doc_snippets = ' '.join(doc_sents)
    else:
        doc_snippets = [' '.join(doc_sents[i:i + 4]) for i in range(len(doc_sents) - 3)]

    unigram_snippets = []
    bigram_snippets = []
    for snip in doc_snippets:
        cnt_u, cnt_b = 0, 0
        for unigr in uni_tweet_tokens:
            if unigr in snip: cnt_u += 1
        overlap_score_u = cnt_u / len(uni_tweet_tokens)
        if overlap_score_u >= OVERLAP_THRESHOLD:
            unigram_snippets.append([snip, overlap_score_u])
        for bigr in bi_tweet_tokens:
            if bigr in snip: cnt_b += 1
        overlap_score_b = cnt_b / len(bi_tweet_tokens)
        if overlap_score_b >= OVERLAP_THRESHOLD:
            bigram_snippets.append([snip, overlap_score_b])

    # Unigrams<Snippets<text, score>>, Bigrams<Snippets<text, score>>
    if len(unigram_snippets) > 0:
        print('Unigrams: {}; bigrams: {}; UrL: {}'.format(len(unigram_snippets), len(bigram_snippets), url))

    return {'unigrams': unigram_snippets,
            'bigrams': bigram_snippets,
            'url': url
            }


def create_user_file(user):
    with open(DIR + 'user_tweet_web_search/' + 'user_' + str(user.user_id) + '.json', 'w') as out_file:
        out_file.write(json.dumps(user.__dict__, default=datetime_converter) + '\n')


def store_search_result(user_id, tweet):
    with open(DIR + 'user_tweet_web_search/' + 'user_' + str(user_id) + '.json', 'a') as out_file:
        out_file.write(json.dumps(tweet, default=datetime_converter) + '\n')


def query_manager(user):
    if user.user_id in pre_crawled_files:
        # Get user file, skip tweets that have been searched for already
        user = skip_pre_searched_tweets(user)
    if len(user.tweets) == 0:
        return
    get_bing_documents_for_tweet(user)
    return


def main():
    global pre_crawled_files
    wn.ensure_loaded()
    pre_crawled_files = [user_file for user_file in glob.glob(DIR + 'user_tweet_web_search/' + 'user_*.json') if
                         'user_' in user_file]
    pre_crawled_files = list(
        map(int, [user_file[user_file.rfind('_') + 1:user_file.rfind('.')] for user_file in pre_crawled_files]))
    users = get_data()
    # Can we parallelize this pls?
    Parallel(n_jobs=NUM_CORES, backend="threading")(delayed(query_manager)(user) for user in users)
    #for user in users: query_manager(user)


if __name__ == "__main__":
    main()
