import json, glob, os, sys

sys.path.insert(0, os.path.dirname(__file__) + '../../2_helpers')
from decoder import decoder

DIR = os.path.dirname(__file__) + '../../../5_Data/'
def get_data():
    fact_file = glob.glob(DIR + 'facts.json')[0]
    transactions_file = glob.glob(DIR + 'factTransaction.json')[0]
    facts = json.load(open(fact_file), object_hook=decoder)
    transactions = json.load(open(transactions_file), object_hook=decoder)
    transactions = sorted(transactions, reverse=False, key=lambda t: t.timestamp)
    print(transactions[0].timestamp)
    print(transactions[1].timestamp)
    print(transactions[2].timestamp)
    return facts, transactions

def get_t_text(f, tr):
    for i, t in enumerate(tr):
        if f.hash == t.fact:
            f.text = t.text
            break
        if i == len(tr)-1: return
    f.text = f.text.strip()
    if len(f.text) == 0:
        get_t_text(f, tr[1:])

facts, transactions = get_data()
for f in facts:
    get_t_text(f, transactions)

[facts.remove(f) for f in facts if len(f.text) == 0]
print(len(facts))


with open(DIR + 'facts.json', 'w') as out_file:
    out_file.write(json.dumps([f.__dict__ for f in facts]))
