import re, os, glob, sys, json
import pandas as pd
import hashlib
import numpy as np
import requests
from joblib import Parallel, delayed
import multiprocessing
from urllib.parse import urlparse
from threading import Thread
import http.client, sys
from queue import Queue
import queue
from collections import Counter
from bs4 import BeautifulSoup
import math

concurrent = 30

num_cores = multiprocessing.cpu_count()
num_jobs = round(num_cores * 3 / 4)
SERVER_RUN = True

DIR = os.path.dirname(__file__) + '../../3_Data/'


OVERLAP_THRESHOLD = 0.4
query_dir = DIR + "user_tweet_query_mod/"
out_dir = DIR + "user_docs/"

def get_data():
    query_files = glob.glob(query_dir + '*_search_results.csv')
    if sys.argv[1]:
        file_preset = sys.argv[1]
        query_files = glob.glob(query_dir + file_preset + '*_search_results.csv')
    prev_snippet_files = glob.glob(out_dir + '*.json')
    prev_snippet_files = [int(snippet[snippet.rfind('/') + 1:snippet.rfind('_snippets')]) for snippet in
                          prev_snippet_files]

    if SERVER_RUN:
        query_files = sorted(query_files, reverse=False, key=lambda x: str(x[x.rfind('/') + 1:x.rfind('_search')]))
    else:
        query_files = sorted(query_files, reverse=True, key=lambda x: str(x[x.rfind('/') + 1:x.rfind('_search')]))

    print('Getting Snippets for {} users'.format(len(query_files)))
    if len(query_files) < 10: print('WRONG DIR?')
    for qfile in query_files:
        userId = int(qfile[qfile.rfind('/') + 1:qfile.rfind('_search')])
        if userId in prev_snippet_files:
            print('Skipping user {}'.format(userId))
            continue

        df = pd.read_csv(qfile, index_col=False)
        df['userId'] = int(qfile[qfile.rfind('/') + 1:qfile.rfind('_search')])
        yield [df, int(qfile[qfile.rfind('/') + 1:qfile.rfind('_search')])]


def parallel_retrieval(urls):
    def doWork():
        while True:
            url = q.get()
            status, url = getStatus(url)
            responses[url] = status
            q.task_done()

    def getStatus(ourl):
        try:
            conn = requests.get(ourl, timeout=5)
            return conn, ourl
        except:
            return "error", ourl

    responses = {}
    q = Queue(concurrent * 2)
    for i in range(concurrent):
        try:
            t = Thread(target=doWork)
            t.daemon = True
            t.start()
        except Exception as e:
            continue
    try:
        for url in urls:
            q.put(url.strip())
        q.join()
    except KeyboardInterrupt:
        sys.exit(1)
    # print(Counter([r.status_code for r in responses.values()]))
    while not q.empty():
        try:
            q.get(False)
        except Empty:
            continue
    del (q)
    return responses


def format_url(url):
    if url is None: return None
    if len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url)) < 1:
        print("Bad URL: {}".format(url))
        return None
    if 'http' not in url[:5]: url = 'http://' + url
    return url


def get_web_doc(url, urlcontent):
    try:
        # html_document = urllib.request.urlopen(url)
        html_document = urlcontent
        soup = BeautifulSoup(html_document.text, 'lxml')

        for script in soup(["script", "style"]):
            script.decompose()
        # get text
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '. '.join(chunk for chunk in chunks if chunk).lower()
        # print("Sucess: {}".format(url))
        text = text.replace('"', "'").replace('\n', '. ')
        return url, text
    except Exception as e:
        # print('Parsing error: {}, for url: {}'.format(e, url))
        return None


def get_tweet_search_results(df, userId):
    print("Working on USER {} with {} entries".format(userId, df.shape))
    if df.shape[0] < 1: return
    if 'link' not in df.columns or 'query' not in df.columns: print('DF EMPTY!!!'); return
    df.drop(['domain', 'effective_query', 'visible_link', 'num_results_for_query', 'num_results', 'link_type',
             'page_number', 'scrape_method', 'status', 'snippet', 'title', 'requested_by', 'search_engine_name',
             'no_results'], axis=1, inplace=True)

    df['query'].replace('', np.nan, inplace=True)
    df.dropna(subset=['query'], inplace=True)

    df['link'] = df['link'].map(lambda x: format_url(x))

    df_range = np.array_split(df, math.ceil((1.0*df.shape[0])/1000))
    print("Splitted into {} subsets".format(len(df_range)))
    for df in df_range:
        print("Querying web docs")
        urls = df['link'].tolist()
        url_contents = parallel_retrieval(urls)

        print("Parsing contents")
        try:
            url_text = Parallel(n_jobs=num_jobs)(delayed(get_web_doc)(x, url_contents[x]) for x in url_contents)
        except Exception as e:
            url_text = [get_web_doc(x, url_contents[x]) for x in url_contents]
        url_text = {unit[0]: unit[1] for unit in url_text if unit is not None}
        df['content'] = df['link'].map(lambda x: url_text[x] if x in url_text else '')

        df['content'].replace('', np.nan, inplace=True)
        df.dropna(subset=['content'], inplace=True)

        df['hash'] = df['query'].map(lambda query: "" if query is None else hashlib.md5(query.encode()).hexdigest())
        if 'content' not in df:
            print("%%%%%%%%%%%%%%%\nCONTENT NOT IN DF\n%%%%%%%%%%%%%%%%")
        print("Finished {} with {} entries".format(userId, df.shape))
        with open(out_dir + str(userId) + '_snippets.json', 'a') as f:
            f.write(df.to_json(orient='records'))
        del(df)


dfs = get_data()
for df in dfs:
    get_tweet_search_results(df[0], df[1])