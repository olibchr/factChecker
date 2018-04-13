import csv
import datetime
import json, glob, time, random
import sys, os

DIR = os.path.dirname(__file__) + '../../3_Data/'

query_files = glob.glob(DIR + 'user_tweet_query/*.csv')

for q_file in query_files:
    print(q_file)
    if sum(1 for line in open(q_file)) <= 1: continue
    with open(q_file, 'r') as f:
        reader = csv.reader((line.replace('\0','') for line in f), delimiter=',', quotechar='"')
        file_headers = next(reader)
        file = []
        for row in reader:
            file.append(row)

    for idx, row in enumerate(file):
        for f_idx, field in enumerate(row):
            file[idx][f_idx] = row[f_idx].replace('\n', ' ').replace('"', "'")

    outdir = DIR + 'user_tweet_query_mod/'
    outfile = q_file[q_file.rfind('/'):]
    with open(outdir + outfile, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(file_headers)
        for row in file:
            writer.writerow(row)