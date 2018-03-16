import glob
import csv


DIR = '/Users/oliverbecher/Google_Drive/0_University_Amsterdam/0_Thesis/3_Data/Qaz/'
TRANSACTIONS = []
FACTS = []

def data_extraction():
    threads = glob.glob(DIR + '*.txt')
    for topic in threads:
        tweets = []
        with open(topic) as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                if len(line) < 1 or (line[-1] != '12' and line[-1] != '11'):
                    continue
                tweets.append(line)
        TRANSACTIONS.extend(tweets)
        FACTS.append(topic)
    #print(TRANSACTIONS)
    print(len([t for t in TRANSACTIONS if t[-1] == '11']))


data_extraction()


