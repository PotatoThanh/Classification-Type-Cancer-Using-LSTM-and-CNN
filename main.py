import numpy as np
import tensorflow as tf
import keras
from keras.layers import *

from data.preprocess_data import preprocess_data
from model.network import my_model
import config

# CONSTANT
LEARNING_RATE = config.learning_rate
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs

# Load data
file_path = 'data.npy'
y_train, x_train_feature, x_train_ref, x_train_tumor,\
y_valid, x_valid_feature, x_valid_ref, x_valid_tumor,\
y_test, x_test_feature, x_test_ref, x_test_tumor = preprocess_data(file_path)

print('############ Shape ############')
print y_train.shape, x_train_feature.shape, x_train_ref.shape, x_train_tumor.shape
print y_valid.shape, x_valid_feature.shape, x_valid_ref.shape, x_valid_tumor.shape
print y_test.shape, x_test_feature.shape, x_test_ref.shape, x_test_tumor.shape

# Input shape
feature_shape = (6, 1)
ref_shape = (96,)
tumor_shape = (82,)
num_class = 6

# Create model
model = my_model(feature_shape, ref_shape, tumor_shape, num_class)

# Compile model with Adam optimizer
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(LEARNING_RATE),
              metrics=['accuracy'])

# call back function
cb_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=BATCH_SIZE,
                            write_graph=True, write_grads=True, write_images=False,
                            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
cb_ckpt = keras.callbacks.ModelCheckpoint('./checkpoint/weights.{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', verbose=1,
                                          save_best_only=True, save_weights_only=False,
                                          mode='auto', period=1)

# train model
model.fit({'feature': x_train_feature, 'ref': x_train_ref, 'tumor': x_train_tumor},
          {'logits': y_train},
          shuffle=True, epochs=EPOCHS, batch_size=EPOCHS, callbacks=[cb_tensorboard, cb_ckpt],
          validation_data=({'feature': x_valid_feature, 'ref': x_valid_ref, 'tumor': x_valid_tumor},
                           {'logits': y_valid}))

# evaluate model
results = model.evaluate({'feature': x_train_feature, 'ref': x_train_ref, 'tumor': x_train_tumor},
               {'logits': y_train},
               verbose=1, batch_size=BATCH_SIZE)

print('Loss: ' + str(results[0]))
print('Accuracy: ' + str(results[1]))