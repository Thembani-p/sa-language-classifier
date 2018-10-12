#!/bin/bash

# run this from your root directory
# create data folder
mkdir data

# setup virtualenv
sudo apt-get install virutalenv

virtualenv --no-site-packages -p python3 sa_lang_env
source sa_lang_env/bin/activate
pip install -r setup/requirements.txt
