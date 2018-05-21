

class Transaction:
    def __init__(self, sourceId, id, user_id, fact, timestamp, stance=None, weight=None, text=None):
        # <sourceId, id, user_id, fact, timestamp, stance, weight, text>
        self.sourceId = sourceId
        self.id = id
        self.user_id = user_id
        self.fact = fact
        self.timestamp = timestamp
        self.stance = stance
        self.weight = weight
        self.text = text
