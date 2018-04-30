class User():
    def __init__(self, user_id, tweets=None, fact=None, transactions=None, credibility=None, controversy=None,
                 features=None, was_correct=None, snippets=None, avg_time_to_retweet=None):
        # <user_id, tweets, fact, transactions, credibility, controversy, features, was_correct, snippets>
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
        self.snippets = snippets
        # Avg time in minutes
        self.avg_time_to_retweet = avg_time_to_retweet
