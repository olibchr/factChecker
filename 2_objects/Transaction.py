

class Transaction:
    def __init__(self, sourceId, id, user_id, fact, timestamp, stance, weight, text):
        # <sourceId, id, user_id, fact, timestamp, stance, weight>
        self.sourceId = sourceId
        self.id = id
        self.user_id = user_id
        self.fact = fact
        self.timestamp = timestamp
        self.stance = stance
        self.weight = weight
        self.text = text
