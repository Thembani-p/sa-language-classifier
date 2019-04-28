

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Multiply, Reshape, Activation, Dropout, Conv2D
from keras.layers import Conv1D, Input, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Lambda, LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.embeddings import Embedding

def simple_ffn(X, y):

    nb_classes = y.shape[1]

    input = Input(shape=(X.shape[1],))

    x = Dense(128)(input)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = BatchNormalization()(x)

    attention_probs = Dense(128, activation='softmax', name='attention_probs')(x)
    attention_mul = Multiply(name='attention_mul')([x, attention_probs])

    x = Dense(nb_classes)(attention_mul)
    output = Activation('softmax')(x)

    model = Model(input, output)

    optimizer = Adam(lr=0.5e-4,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08,
                        decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def parameter_ffn(X, y, layers = [128],
                        activations = ['relu'],
                        dropout = [0.15],
                        attention = 128):

    # allow for mixed length parameters if all are 1 and some are k
    max_len = max(len(layers), len(activations), len(dropout))

    if len(layers) == 1:
        layers = layers*max_len
    if len(activations) == 1:
        activations = activations*max_len
    if len(dropout) == 1:
        dropout = dropout*max_len

    assert len(layers) == len(activations) == len(dropout), 'your parameters should have the same length or a length of 1'
    assert type(attention) != 'int', 'the attention layers should be given as an integer'
    # base network
    nb_classes = y.shape[1]

    input = Input(shape=(X.shape[1],))

    x = Dense(layers[0])(input)
    x = Activation(activations[0])(x)
    x = Dropout(dropout[0])(x)
    x = BatchNormalization()(x)

    # dynamic network
    for l, layer in enumerate(layers[1::]):
        x = Dense(layer)(x)
        x = Activation(activations[l])(x)
        x = Dropout(dropout[l])(x)
        x = BatchNormalization()(x)

    attention_probs = Dense(attention, activation='softmax', name='attention_probs')(x)
    attention_mul = Multiply(name='attention_mul')([x, attention_probs])

    x = Dense(nb_classes)(attention_mul)
    output = Activation('softmax')(x)

    model = Model(input, output)

    optimizer = Adam(lr=0.5e-4,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08,
                        decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def parameter_ffn_seq(input_shape=5, classes=2, layers = [128],
                        activations = ['relu'],
                        dropout = [0.15]):

    # allow for mixed length parameters if all are 1 and some are k
    max_len = max(len(layers), len(activations), len(dropout))

    if len(layers) == 1:
        layers = layers*max_len
    if len(activations) == 1:
        activations = activations*max_len
    if len(dropout) == 1:
        dropout = dropout*max_len

    assert len(layers) == len(activations) == len(dropout), 'your parameters should have the same length or a length of 1'
    # assert type(attention) != 'int', 'the attention layers should be given as an integer'
    # base network

    model = Sequential()

    model.add(Dense(layers[0], input_shape=(input_shape,)))
    model.add(Activation(activations[0]))
    model.add(Dropout(dropout[0]))
    model.add(BatchNormalization())

    # dynamic network
    for l, layer in enumerate(layers[1::]):
        model.add(Dense(layers[l]))
        model.add(Activation(activations[l]))
        model.add(Dropout(dropout[l]))
        model.add(BatchNormalization())

    # attention_probs = Dense(attention, activation='softmax', name='attention_probs')(x)
    # attention_mul = Multiply(name='attention_mul')([x, attention_probs])

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=0.5e-4,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08,
                        decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def sequence_model(X, y, num_unique_symbols):

    nb_classes = y.shape[1]

    input = Input(shape=(X.shape[1],num_unique_symbols))

    x = LSTM(128, return_sequences=True)(input)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = BatchNormalization()(x)

    x = LSTM(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = BatchNormalization()(x)

    attention_probs = Dense(128, activation='softmax', name='attention_probs')(x)
    attention_mul = Multiply(name='attention_mul')([x, attention_probs])

    x = Dense(nb_classes)(attention_mul)
    output = Activation('softmax')(x)

    model = Model(input, output)

    optimizer = Adam(lr=0.5e-4,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08,
                        decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
