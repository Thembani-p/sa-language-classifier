

# ------------------------------------------------------------------------------
# laod packages
# ------------------------------------------------------------------------------
import os
import sys
import json
import pickle

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
    X_train, X_test, y_train, y_test = train_test_split(flat_corpora,
                                                        flat_labels,
                                                        test_size=0.25,
                                                        random_state=123)

    X_train, y_train, pipeline_instance = prepare_training_data(X_train, y_train)

    print(X_train.shape, y_train.shape, pipeline_instance)

    print('Saving pipeline \n')
    with open(os.path.join(DATA_PATH, 'pipeline_instance.pickle'),'wb') as datafile:
        pickle.dump(pipeline_instance, datafile)

    print('Train mdoel \n')
    model = simple_ffn(X_train, y_train)
    print(model.summary())

    model.fit(X_train, y_train, epochs=50, validation_split=0.1, verbose=1)

    model.save(os.path.join(MODELS_PATH,'simple_model.h5'))

    X_test = pipeline_instance.fit_transform(X_test)
    y_test = pd.get_dummies(y_test)

    score, accuracy = model.evaluate(X_test, y_test)
    print('Model test accuracy', accuracy.round(4))
