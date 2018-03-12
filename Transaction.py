

class Transaction:
    def __init__(self, sourceID, id, user_id, fact, timestamp, stance, weight):
        self.sourceId = sourceID
        self.id = id
        self.user_id = user_id
        self.fact = fact
        self.timestamp = timestamp
        self.stance = stance
        self.weight = weight
