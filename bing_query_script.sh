#!/bin/bash
if [ ! -d ./userCredibility ]; then
  mkdir -p ./userCredibility;
fi
cd userCredibility/
message="Downloading files"

# Download all previous data
DATAFILE="https://goo.gl/uH17n1"
TMPFILE="3_Data.zip"
PWD=`pwd`
# wget $DATAFILE -O $TMPFILE; unzip -d $PWD $TMPFILE; rm $TMPFILE &

# Download Repo
TMPFILEREPO="factChecker-master.zip"
REPO="https://github.com/olibchr/factChecker/archive/master.zip"
PWD=`pwd`
wget $REPO -O $TMPFILEREPO
unzip -d $PWD $TMPFILEREPO
rm $TMPFILEREPO
cd factChecker-master/

# Install dependencies
# Setup VENV
pip install --user virtualenv
mkdir venv
python3 -m venv venv/
source venv/bin/activate
pip3 install --upgrade pip --user
pip3 install --user -r requirements.txt

# Get right AIOHTTP Version
pip3 uninstall aiohttp
cd ../
mkdir 9_packages/
cd 9_packages
wget https://github.com/aio-libs/aiohttp/archive/4v0.21.6.zip -O aiohttp.zip
unzip aiohttp.zip
rm aiohttp.zip
cd aiohttp-4v0.21.6/
python3 setup.py install --user
cd ..

# Install googlescraper
wget https://github.com/olibchr/GoogleScraper/archive/master.zip -O GoogleScraper.zip
unzip GoogleScraper.zip
rm GoogleScraper.zip
cd GoogleScraper-master/
python3 setup.py install --user
cd ../..
ls
cd factChecker-master/0_data_retrieval/
python3 get_tweet_search_results.py 3