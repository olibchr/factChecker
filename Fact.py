import hashlib

class Fact():
    def __init__(self, rumor_type, topic, text, true, proven_false, is_turnaround):
        # <RUMOR_TPYE, HASH, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND>
        self.rumor_type = rumor_type
        self.hash = hashlib.md5(text.encode()).hexdigest()
        self.topic = topic
        self.text = text
        self.true = true
        self.proven_false = proven_false
        self.is_turnaround = is_turnaround
    def set_triple_set(self, triples):
        self.triples = triples
