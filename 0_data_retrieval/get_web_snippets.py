from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import lit, col, udf, explode
from pyspark.sql.types import *
import re, os
import hashlib
import requests

TEST_RUN = False

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

conf = SparkConf().setAppName("tweet_web_crawl")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

search_engine_files = "3_data/user_tweet_query/*_search_results.csv"
user_files = "3_data/user_tweets/user_*.json"
out_dir = "3_data/user_snippets/"
if TEST_RUN: search_engine_files = DIR + "user_tweet_query/0000000_search_results copy.csv"
if TEST_RUN: user_files = DIR + "user_tweets/user_14723131.json"
if TEST_RUN: out_dir = DIR + 'user_snippets/'


def get_ngram_snippets(tweet_text, web_document, url):
    from nltk.stem import WordNetLemmatizer
    WNL = WordNetLemmatizer()
    # Remove odd characters from tweet, to lowercast and remove stopword. Parse into uni and bigrams
    tweet_text = re.sub(r'[^a-z0-9]', ' ', tweet_text.lower())

    # Lemamtization of tweet, remove stopwords, put into uni and bigrams
    uni_tweet_tokens = [WNL.lemmatize(i) for i in nltk.word_tokenize(tweet_text) if i not in NLTK_STOPWORDS]
    bi_tweet_tokens = [' '.join(uni_tweet_tokens[i:i + 2]) for i in range(len(uni_tweet_tokens) - 1)]
    if len(uni_tweet_tokens) == 0:
        return {'unigrams': [],
                'bigrams': [],
                'url': url
                }
    if len(bi_tweet_tokens) == 0: bi_tweet_tokens = uni_tweet_tokens

    # Split doc into senctences, lemmatize, remove stopwords and to lowercase
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


def get_web_doc(url):
    try:
        # html_document = urllib.request.urlopen(url)
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
    return hashlib.md5(tweet_text.encode()).hexdigest()


def get_tweet_search_results(df):
    get_web_doc_udf = udf(get_web_doc, StringType())
    get_ngram_snippets_udf = udf(get_ngram_snippets, MapType(StringType(), ArrayType(ArrayType(StringType()))))
    get_hash = udf(lambda query: hashlib.md5(query.encode()).hexdigest(), StringType())

    df = df.withColumn("content", get_web_doc_udf(df['link']))
    df = df.withColumn("snippets", get_ngram_snippets_udf(df['query'], df['content'], df['link']))
    df = df.withColumn("hash", get_hash(df['query']))
    df = df.drop('content')
    df.show()
    return df


def get_hash_for_user_tweets(df_users):
    get_hash = udf(extract_query_term, StringType())
    df_users = df_users.select(explode('tweets').alias("tweet"), 'user_id', 'was_correct', 'features', 'credibility',
                               'transactions', 'fact')
    df_users.show()
    df_users = df_users.withColumn("hash", get_hash(df_users['tweet']))
    return df_users


df = sqlContext.read.load(search_engine_files,
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')

df_users = sqlContext.read.json(DIR + user_files)

df = df.drop('domain', 'effective_query', 'visible_link', 'num_results_for_query', 'num_results', 'link_type',
             'page_number', 'scrape_method', 'status', 'snippet', 'title', 'requested_by', 'search_engine_name',
             'no_results', )
df = get_tweet_search_results(df)
df_users = get_hash_for_user_tweets(df_users)

df1 = df.alias('df1')
df2 = df_users.alias('df2')
df = df1.join(df2, df1.hash == df2.hash).select('df1.*', 'df2.tweet', 'df2.user_id', 'df2.was_correct', 'df2.features',
                                                'df2.credibility', 'df2.transactions', 'df2.fact')
df.show()
df.write.mode('overwrite').partitionBy("hash").format('json').save(out_dir)
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
