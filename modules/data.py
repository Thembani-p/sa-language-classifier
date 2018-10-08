

import json

import numpy as np
import pandas as pd

import nltk

from keras.preprocessing.text import one_hot, Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import CountVectorizer,  TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Language codes

# isiNdebele - nr
# isiXhosa   - xh
# isiZulu    - zu
# sePedi     - nso
# seSotho    - st
# seTswana   - tn
# siSwati    - ss
# tshiVenda  - ve
# xiTsonga   - ts

LANG_KEY = ['nr','xh','zu','nso','st','tn','ss','ve','ts']

def raise_one_level(nested):
    result = []
    for i in nested:
        result += i
    return result

def flatten_data(single_corpora, language_key=LANG_KEY):
    # prepare text
    single_corpora_sentences = [[' '.join(j) for j in i] for i in single_corpora]
    single_corpora_sentences = [[j for j in i if len(j) > 1] for i in single_corpora_sentences]
    flat_corpora = raise_one_level(single_corpora_sentences)

    # prepare labels
    labels_nested = [[language_key[idx] for j in i] for idx,i in enumerate(single_corpora_sentences)]
    flat_labels = raise_one_level(labels_nested)

    return flat_corpora, flat_labels

def prepare_training_data(flat_corpora, flat_labels):
    text_pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer='char', ngram_range=(2,2))),
        ('tfidf', TfidfTransformer())
    ])

    pipeline_instance = text_pipeline.fit(flat_corpora)
    vectors = pipeline_instance.transform(flat_corpora)

    vocab = pipeline_instance.steps[0][1].get_feature_names()

    document_matrix = pd.DataFrame(vectors.toarray(),columns=vocab)
    labels = pd.get_dummies(flat_labels)

    return document_matrix, labels, pipeline_instance

def prepare_test_data(flat_corpora, flat_labels, pipeline_instance):
    vocab = pipeline_instance.steps[0][1].get_feature_names()
    vectors = pipeline_instance.transform(flat_corpora)

    document_matrix = pd.DataFrame(vectors.toarray(),columns=vocab)
    labels = pd.get_dummies(flat_labels)

    return document_matrix, labels
