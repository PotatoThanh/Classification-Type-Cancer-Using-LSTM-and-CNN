import numpy as np
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K

def my_model(feature_shape, ref_shape, tumor_shape, num_class):
    # ref LSTM
    input_ref = Input(ref_shape, name='ref')

    with tf.name_scope('Embedding-Conv1D-LSTM-1'):
        ref = Embedding(5, 64, input_length=ref_shape[0])(input_ref)

        ref = Conv1D(128, 4, activation='relu')(ref)
        ref = MaxPool1D(2)(ref)

        ref = LSTM(100)(ref)
        ref = Dropout(0.35)(ref)

    # tumor LSTM
    input_tumor = Input(tumor_shape, name='tumor')

    with tf.name_scope('Embedding-Conv1D-LSTM-2'):
        tumor = Embedding(5, 64, input_length=tumor_shape[0])(input_tumor)

        tumor = Conv1D(128, 4, activation='relu')(tumor)
        tumor = MaxPool1D(2)(tumor)

        tumor = LSTM(100)(tumor)
        tumor = Dropout(0.35)(tumor)

    # feature CNN
    input_feature = Input(feature_shape, name='feature')

    with tf.name_scope('Conv1D-Dense-3'):
        feature = Conv1D(128, 4, activation='relu')(input_feature)
        feature = MaxPool1D(2)(feature)

        feature = Flatten()(feature)

        feature = Dense(100, activation='relu')(feature)
        feature = Dropout(0.35)(feature)

    # concatenate
    cat = concatenate([feature, ref, tumor])

    with tf.name_scope('Dense'):
        cat = Dense(150, activation='relu')(cat)
        cat = Dropout(0.3)(cat)

        cat = Dense(50, activation='relu')(cat)
        cat = Dropout(0.2)(cat)

    outputs = Dense(num_class, activation='softmax', name='logits')(cat)

    model = Model(inputs=[input_feature, input_ref, input_tumor],
                  outputs=[outputs])

    return model