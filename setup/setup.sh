#!/bin/bash

# run this from your root directory
# clone repo
# git clone https://github.com/Thembani-p/sa-language-classifier.git
# cd sa-language-classifier

# create data folder
mkdir data
wget https://storage.googleapis.com/sa-languages/single_corpora.json -P data

# setup virtualenv
sudo apt-get update
sudo apt-get install python3-pip -y
# sudo apt-get install wget -y
sudo pip3 install virtualenv

virtualenv --no-site-packages -p python3 sa_lang_env
source sa_lang_env/bin/activate
pip install -r setup/requirements.txt

python scripts/train_simple_model.py
