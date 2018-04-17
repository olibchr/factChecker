class User():
    def __init__(self, user_id, tweets=None, fact=None, transactions=None, credibility=None, controversy=None,
                 features=None, was_correct=None, snippets=None):
        # <user_id, tweets, fact, transactions, credibility, controversy, features, was_correct, snippets>
        self.user_id = user_id
        # Tweets <text, created_at, *quoted_status>
        self.tweets = tweets
        self.fact = fact
        self.transactions = transactions
        self.credibility = credibility
        self.controversy = controversy
        self.features = features
        self.was_correct = was_correct
        self.snippets = snippets
