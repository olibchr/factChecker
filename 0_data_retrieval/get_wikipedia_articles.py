import textrazor, json, warnings, sys, os, glob
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__) + '../2_objects')
from Fact import Fact
from Transaction import Transaction

warnings.filterwarnings("ignore", category=DeprecationWarning)

DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/'
# DIR = '/var/scratch/obr280/0_Thesis/3_Data/'
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

def get_wikilinks(facts):
    textrazor.api_key = "f87456a08da3eff12e62ebdb2bcf2a8be4baaeb4b79be19fce12f770"
    client = textrazor.TextRazor(extractors=["entities", "topics"])
    for fact in facts:
        response = client.analyze(fact.text)
        json_response = response.json
        print(json_response)
        exit()


def main():
    facts, transactions = get_data()
    get_wikilinks(facts)
if __name__ == "__main__":
    main()

