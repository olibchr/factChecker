class User():
    def __init__(self, user_id, tweets=None, fact=None, transactions=None, credibility=None, controversy=None,
                 features=None, was_correct=None, snippets=None, avg_time_to_retweet=None, sent_tweets_density=None, sent_tweets_avg=None, target_stance=None, target_certainty=None):
        # <user_id, tweets, fact, transactions, credibility, controversy, features, was_correct, snippets, avg_time_to_retweet>
        self.user_id = user_id
        # Tweets <text, created_at, reply_to, retweets, favorites, *quoted_status<created_at, text>>
        self.tweets = tweets
        self.fact = fact
        self.transactions = transactions
        self.credibility = credibility
        self.controversy = controversy
        # features {name, location, followers, friends, description, created_at, verified, statuses_count, lang}
        self.features = features
        self.was_correct = was_correct
        # Unigrams and bigrams with webpages
        self.snippets = snippets
        # Avg time in minutes
        self.avg_time_to_retweet = avg_time_to_retweet
        # Sentiment of tweets
        self.sent_tweet_avg = sent_tweets_avg
        # Buckets are np.arange(-1,1.1,0.2)
        self.sent_tweet_density = sent_tweets_density
        # Annotated data
        self.target_stance = target_stance
        self.target_certainty = target_certainty
