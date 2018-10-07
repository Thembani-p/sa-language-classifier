

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
