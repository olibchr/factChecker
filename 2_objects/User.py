class User():
    def __init__(self, user_id, tweets=None, fact=None, transactions=None, credibility=None, controversy=None, features=None):
        # <user_id, tweets, fact, transactions, credibility, controversy>
        self.user_id = user_id
        # Tweets <text, created_at, *quoted_status>
        self.tweets = tweets
        self.fact = fact
        self.transactions = transactions
        self.credibility = credibility
        self.controversy = controversy
        self.features = features
