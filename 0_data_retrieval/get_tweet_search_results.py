from datetime import datetime
import json, glob, time, random
import sys, os
from six.moves import urllib
sys.path.insert(0, os.path.dirname(__file__) + '../2_helpers')
import re, nltk
from User import User
from bs4 import BeautifulSoup
from GoogleScraper import scrape_with_config, GoogleSearchError

SERVER_RUN = True

DIR = os.path.dirname(__file__) + '../../3_Data/'

OVERLAP_THRESHOLD = 0.4

watchdog = 0


def make_web_query(keywords, userid):
    config = {
        'use_own_ip': 'True',
        'keywords': keywords,
        'search_engines': ['bing'],
        'num_pages_for_keyword': 1,
        'scrape_method': 'http-async',
        'do_caching': 'True',
        'output_filename': '../../3_Data/user_tweet_query/' + str(userid) + '_search_results.csv'
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
        results.append({'query': keywords[idx], 'links': links, 'serp': str(serp)})
    return results


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
    preset_first_occurence = [idx for idx, uf in enumerate(user_files) if 'user_' + user_file_preset in uf][0]
    user_files = user_files[preset_first_occurence:]
    print('Getting Search Results for {} users'.format(len(user_files)))
    if SERVER_RUN: user_files = sorted(user_files, reverse=False)
    else: user_files = sorted(user_files, reverse=True)

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

    search_terms = []
    tweets = sorted(user.tweets, key=lambda t: t['created_at'])
    if len(tweets) > 300: tweets = tweets[:300]
    for tweet in tweets:
        # user.tweets <text, created_at, quoted_status>
        # if not is_tweet_fact (tweet): continue
        # If tweet contains link to another tweet, link should be resolved and tweet parsed..
        if not is_tweet_fact(tweet):
            # print('Gonna skip: {}'.format(tweet['text']))
            continue
        query_term = extract_query_term(tweet)
        tweet['search_instance'] = len(search_terms)
        if is_tweet_fact(tweet):
            search_terms.append(query_term)

    print('Searching for {} terms'.format(len(search_terms)))
    return tweets, make_web_query(search_terms, user.user_id)


def query_manager(user):
    if user.user_id in pre_crawled_files:
        # Get user file, skip tweets that have been searched for already
        print('Skipping User {}'.format(user.user_id))
        return
    if not user.tweets or len(user.tweets) == 0:
        return

    # Get documents from bing search related to tweet
    try:
        get_bing_documents_for_tweet(user)
    except Exception as e:
        print('Resting for a bit: {}'.format(e))
        time.sleep(random.randrange(10,20))
        try:
            get_bing_documents_for_tweet(user)
        except Exception as e:
            return

    print('Sucessfully processed user: {}'.format(user.user_id))


def main():
    global pre_crawled_files
    pre_crawled_files = glob.glob('../../3_Data/user_tweet_query/' + '*_search_results.csv')
    pre_crawled_files = list(
        map(int, [user_file[user_file.rfind('/') + 1:user_file.rfind('_search')] for user_file in pre_crawled_files]))
    users = get_data()
    for user in users: query_manager(user)


# noinspection PyPackageRequirements
if __name__ == "__main__":
    global user_file_preset
    user_file_preset = sys.argv[1]
    main()
