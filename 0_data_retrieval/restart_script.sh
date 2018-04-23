#!/bin/bash
kill -9 $1
python3 /var/scratch/obr280/0_Thesis/2_Code/0_data_retrieval/get_web_docs.py $2
