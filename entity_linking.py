import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
import time
import numpy as np
import nltk
import spotlight
import warnings, json, glob
from Fact import Fact
from Transaction import Transaction

warnings.filterwarnings("ignore", category=DeprecationWarning)

DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/'
SHOW_PLOTS = False

def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()

def fact_decoder(obj):
    # <RUMOR_TPYE, HASH, TOPIC, TEXT, TRUE, PROVEN_FALSE, TURNAROUND, SOURCE_TWEET>
    return Fact(obj['rumor_type'], obj['topic'], obj['text'], obj['true'], obj['proven_false'],
                obj['is_turnaround'], obj['source_tweet'], hash=obj['hash'])


def get_data():
    facts = json.load(open(glob.glob(DIR + 'facts.json')[0]), object_hook=fact_decoder)
    transactions = json.load(open(glob.glob(DIR + 'factTransaction.json')[0]))
    return facts, transactions


def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names


def preprocess(facts):
    for fact in facts:
        sentences = nltk.sent_tokenize(fact.text)
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
        chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

        entity_names = []
        for tree in chunked_sentences:
            entity_names.extend(extract_entity_names(tree))
        fact.set_entities(entity_names)


def get_linked_entities_spotlight(facts):
    for fact in facts:
        print(fact.text)
        try:
            annotations = spotlight.annotate('http://model.dbpedia-spotlight.org/en/annotate',fact.text, confidence=0.4, support=20)
        except spotlight.SpotlightException as e:
            print('No annotaions')
            continue
        print([a['surfaceForm'] for a in annotations])
        fact.set_entities(annotations)


def store_result(facts):
    with open(DIR + 'facts_annotated.json', 'w') as out_file:
        out_file.write(json.dumps([f.__dict__ for f in facts], default=datetime_converter))



def main():
    facts, transactions = get_data()
    #preprocess(facts)
    get_linked_entities_spotlight(facts)
    store_result(facts)


if __name__ == "__main__":
    main()
