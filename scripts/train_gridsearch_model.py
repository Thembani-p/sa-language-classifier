

# ------------------------------------------------------------------------------
# laod packages
# ------------------------------------------------------------------------------
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------
MODULES_PATH = '../modules'
MODELS_PATH = '../models'
DATA_PATH = '../data'

sys.path.append(MODULES_PATH)

sys.path.append(MODULES_PATH)
from data import flatten_data, prepare_training_data, prepare_test_data, \
                    raise_one_level
from models import parameter_ffn_seq


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run dpr for streetbees datasets.')
    parser.add_argument('-n','--neurons', required=True,
                        type=int,
                        metavar="64",
                        help='Full path of the file the dqr will be run on.')
    parser.add_argument('-t','--test', required=False,
                        default=0.75,
                        metavar="0.25",
                        help='Test split for running grid search')
    parser.add_argument('-c','--cpu', required=True,
                        type=int,
                        metavar="3",
                        default=3,
                        help='Cpu cores for parallelism.')

    args = parser.parse_args()

    print('Gridearch of {} neurons with {} test size'.format(args.neurons, args.test))

    print('Read data \n')
    with open(os.path.join(DATA_PATH,'sentences.json'),'r') as datafile:
        sentences = json.load(datafile)

    sentences_flat = raise_one_level(sentences)
    sentences_df = pd.DataFrame(sentences_flat)

    corpora_train, corpora_test, labels_train, labels_test = train_test_split(
                                                                        sentences_df['body'],
                                                                        sentences_df['class'],
                                                                        test_size=float(args.test),
                                                                        random_state=123)
    print("Training data \n")
    training_data = []
    for i in range(1,2):
        print(i+1)

        document_matrix, labels, pipeline_instance = prepare_training_data(corpora_train, labels_train, (i,i))
        training_data.append({'document_matrix': document_matrix, 'labels': labels, 'pipeline_instance': pipeline_instance})

    for i in training_data:
        print(i['document_matrix'].shape)

    print("Create and train model \n")
    ffn = KerasClassifier(build_fn=parameter_ffn_seq, verbose=1)

    parameters = {'layers': [],
                   'activations': [['relu']],
                   'dropout': [[0.05], [0.15], [0.25]],
                   'attention': [128],
                  'input_shape': [document_matrix.iloc[0:5].shape[1]],
                  'nb_classes': [labels.iloc[0:5].shape[1]]}

    for j in [args.neurons]:
        for i in range(3):
            parameters['layers'].append([j]*(i+1))

    ffn_grid = GridSearchCV(estimator=ffn, param_grid=parameters, n_jobs=args.cpu, verbose=1)

    grid_result = ffn_grid.fit(document_matrix, labels, epochs=100)

    pd.DataFrame(grid_result.cv_results_).sort_values('mean_test_score', ascending=False).to_csv(os.path.join(MODELS_PATH, 'grid_result_{}.csv'.format(args.neurons)))

    with open(os.path.join(MODELS_PATH, 'grid_result_{}.pickle'.format(args.neurons)),'wb') as datafile:
        pickle.dump(grid_result, datafile)

    print('All done')
