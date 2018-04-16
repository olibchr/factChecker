import re, os, glob, sys, json
import pandas as pd
import hashlib
import numpy as np
import requests
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
num_jobs = 4 #round(num_cores * 3 / 4)
SERVER_RUN = False

DIR = os.path.dirname(__file__) + '../../3_Data/'


def lemmatize(x):
    from nltk.stem import WordNetLemmatizer
    WNL = WordNetLemmatizer()
    return WNL.lemmatize(x)


def get_soup(html_document):
    from bs4 import BeautifulSoup
    return BeautifulSoup(html_document.text, 'lxml')


NLTK_STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                  'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                  'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                  'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                  'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                  'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                  'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                  'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                  'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                  'now']

OVERLAP_THRESHOLD = 0.4
query_dir = DIR + "user_tweet_query_mod/*_search_results.csv"
search_engine_file_preset = sys.argv[1]
out_dir = DIR + "user_snippets/"


# if not SERVER_RUN: search_engine_files = DIR + "user_tweet_query_mod/14*_results.csv"
# if not SERVER_RUN: out_dir = DIR + 'user_snippets/'


def get_data():
    search_engine_files = glob.glob(query_dir)
    preset_first_occurence = [idx for idx, uf in enumerate(search_engine_files) if search_engine_file_preset + '_search' in uf][0]
    query_files = search_engine_files[preset_first_occurence:]
    print('Getting Snippets for {} users'.format(len(query_files)))

    prev_snippet_files = glob.glob(out_dir + '*.json')
    prev_snippet_files = [int(snippet[snippet.rfind('/') + 1:snippet.rfind('_snippets')]) for snippet in
                          prev_snippet_files]

    if SERVER_RUN:
        search_engine_files_subset = sorted(query_files, reverse=False)
    else:
        search_engine_files_subset = sorted(query_files, reverse=True)

    if len(search_engine_files_subset) < 10: print('WRONG DIR?')
    for qfile in query_files:
        userId = int(qfile[qfile.rfind('/') + 1:qfile.rfind('_search')])
        if userId in prev_snippet_files:
            print('Skipping user {}'.format(print(userId)))
            continue

        df = pd.read_csv(qfile, index_col=False)
        df['userId'] = int(qfile[qfile.rfind('/') + 1:qfile.rfind('_search')])
        yield [df, int(qfile[qfile.rfind('/') + 1:qfile.rfind('_search')])]


def get_ngram_snippets(tweet_text, web_document, url):
    from nltk.stem import WordNetLemmatizer
    from nltk import word_tokenize, sent_tokenize
    if web_document is None or tweet_text is None: return None
    WNL = WordNetLemmatizer()
    # Remove odd characters from tweet, to lowercast and remove stopword. Parse into uni and bigrams
    tweet_text = re.sub(r'[^a-z0-9]', ' ', tweet_text.lower())

    # Lemamtization of tweet, remove stopwords, put into uni and bigrams
    uni_tweet_tokens = [WNL.lemmatize(i) for i in word_tokenize(tweet_text) if i not in NLTK_STOPWORDS]
    bi_tweet_tokens = [' '.join(uni_tweet_tokens[i:i + 2]) for i in range(len(uni_tweet_tokens) - 1)]
    if len(uni_tweet_tokens) == 0:
        return {'unigrams': [],
                'bigrams': [],
                'url': url
                }
    if len(bi_tweet_tokens) == 0: bi_tweet_tokens = uni_tweet_tokens

    # Split doc into senctences, lemmatize, remove stopwords and to lowercase
    doc_sents = sent_tokenize(web_document)
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
            # if len(unigram_snippets) > 0:
            # print('Unigrams: {}; bigrams: {}; UrL: {}'.format(len(unigram_snippets), len(bigram_snippets), url))

    return {'unigrams': unigram_snippets,
            'bigrams': bigram_snippets,
            'url': url
            }


def get_web_doc(url):
    if url is None: return None
    if len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url)) < 1:
        print("Bad URL: {}".format(url))
        return None

    try:
        # html_document = urllib.request.urlopen(url)
        if 'http' not in url[:5]: url = 'http://' + url
        html_document = requests.get(url)
        soup = get_soup(html_document)

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


def extract_query_term(tweet):
    if not tweet: return ""
    tweet_text = tweet['text']
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet_text)
    for url in urls:
        if 'https://t.co/' in url:
            tweet_text = tweet_text.replace(url, '')
    if 'quoted_status' in tweet and tweet['quoted_status']:
        tweet_text = tweet_text + ' ' + tweet['quoted_status']
    if tweet_text[:2].lower() == 'rt': tweet_text = tweet_text[2:]
    tweet_text = tweet_text.replace('\0', '').replace('\n', ' ')
    return hashlib.md5(tweet_text.encode()).hexdigest()


def get_tweet_search_results(df, userId):
    print("Working on {} with {} entries".format(userId, df.shape))
    if df.shape[0] < 1: return
    if 'link' not in df.columns or 'query' not in df.columns: print('DF EMPTY!!!'); return
    df.drop(['domain', 'effective_query', 'visible_link', 'num_results_for_query', 'num_results', 'link_type',
             'page_number', 'scrape_method', 'status', 'snippet', 'title', 'requested_by', 'search_engine_name',
             'no_results'], axis=1, inplace=True)

    df['query'].replace('', np.nan, inplace=True)
    df.dropna(subset=['query'], inplace=True)

    df['content'] = df['link'].map(lambda x: get_web_doc(x))

    df['content'].replace('', np.nan, inplace=True)
    df.dropna(subset=['content'], inplace=True)

    df['snippets'] = df.apply(lambda x: get_ngram_snippets(x['query'], x['content'], x['link']), axis=1)

    df['hash'] = df['query'].map(lambda query: "" if query is None else hashlib.md5(query.encode()).hexdigest())

    df.drop('content', axis=1, inplace=True)
    with open(out_dir + str(userId) + '_snippets.json', 'w') as f:
        f.write(df.to_json(orient='records'))


dfs = get_data()

#[get_tweet_search_results(df[0],df[1]) for df in dfs]
Parallel(n_jobs=num_jobs)(delayed(get_tweet_search_results)(df[0], df[1]) for df in dfs)

# root
#  |-- domain: string (nullable = true)
#  |-- effective_query: string (nullable = true)
#  |-- link: string (nullable = true)
#  |-- link_type: string (nullable = true)
#  |-- no_results: string (nullable = true)
#  |-- num_results: string (nullable = true)
#  |-- num_results_for_query: string (nullable = true)
#  |-- page_number: string (nullable = true)
#  |-- query: string (nullable = true)
#  |-- rank: string (nullable = true)
#  |-- requested_at: string (nullable = true)
#  |-- requested_by: string (nullable = true)
#  |-- scrape_method: string (nullable = true)
#  |-- search_engine_name: string (nullable = true)
#  |-- snippet: string (nullable = true)
#  |-- status: string (nullable = true)
#  |-- title: string (nullable = true)
#  |-- visible_link: string (nullable = true)
