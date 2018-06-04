class User():
    def __init__(self, user_id, tweets=None, fact=None, fact_text=None, fact_text_ts=None, transactions=None, credibility=None, controversy=None,
                 features=None, was_correct=None, avg_time_to_retweet=None, sent_tweets_avg=None, stance=None, certainty=None):
        # <user_id, tweets, fact, transactions, credibility, controversy, features, was_correct, snippets, avg_time_to_retweet>
        self.user_id = user_id
        # Tweets <text, created_at, reply_to, retweets, favorites, *quoted_status<created_at, text>>
        self.tweets = tweets
        self.fact = fact
        self.fact_text = fact_text
        self.fact_text_ts = fact_text_ts

        self.transactions = transactions
        self.controversy = controversy
        # features {name, location, followers, friends, description, created_at, verified, statuses_count, lang}
        self.features = features
        self.was_correct = was_correct
        # Avg time in minutes
        self.avg_time_to_retweet = avg_time_to_retweet
        # Sentiment of tweets
        self.sent_tweets_avg = sent_tweets_avg
        # Annotated data
        self.stance = stance
        self.certainty = certainty
