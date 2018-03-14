import hashlib

class Fact():
    def __init__(self, rumor_type, topic, text, true, proven_false, is_turnaround, source_tweet, hash = None):
        # <RUMOR_TPYE, HASH, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND, SOURCE_TWEET>
        self.rumor_type = rumor_type
        self.hash = hashlib.md5(text.encode()).hexdigest() if hash is None else hash
        self.topic = topic
        self.text = text
        self.true = true
        self.proven_false = proven_false
        self.is_turnaround = is_turnaround
        self.source_tweet = source_tweet
        self.entities = []
    def set_triple_set(self, triples):
        self.triples = triples
    def set_entities(self, entities):
        self.entities = entities
