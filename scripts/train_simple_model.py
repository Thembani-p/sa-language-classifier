

# ------------------------------------------------------------------------------
# laod packages
# ------------------------------------------------------------------------------
import os
import sys
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------
MODULES_PATH = '../modules'
MODELS_PATH = '../models'
DATA_PATH = '../data'

sys.path.append(MODULES_PATH)

from data import flatten_data, prepare_training_data, prepare_test_data
from models import simple_ffn




# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    print('Read data \n')
    with open(os.path.join(DATA_PATH,'single_corpora.json'),'r') as datafile:
        single_corpora = json.load(datafile)

    flat_corpora, flat_labels = flatten_data(single_corpora)
    document_matrix, labels, pipeline_instance = prepare_training_data(flat_corpora, flat_labels)

    print(document_matrix.shape, labels.shape, pipeline_instance)
    print('Split data \n')

    X_train, X_test, y_train, y_test = train_test_split(document_matrix,
                                                        labels,
                                                        test_size=0.25,
                                                        random_state=123)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                          y_train,
                                                          test_size=0.25,
                                                          random_state=321)

    print('Train mdoel \n')
    model = simple_ffn(X_train, y_train)
    print(model.summary())

    model.fit(X_train, y_train, epochs=25, validation_data=(X_valid, y_valid), verbose=1)

    model.save(os.path.join(MODELS_PATH,'simple_model.h5'))

    score, accuracy = model.evaluate(X_test, y_test)
    print('Model test accuracy', accuracy)
