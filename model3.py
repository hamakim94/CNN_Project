# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from functions import ready_callbacks

path_q = input('데이터가 현재 폴더위치에 있습니까?[y/n]')

if path_q == 'y': 
  path = os.getcwd()

else:
  path = str(input('경로를 입력하세요'))
print(path)

# 데이터 로딩 
def load_data(path):
  test = pd.read_pickle(path +'/'+ "token_test_data.pkl")
  train = pd.read_pickle(path +'/' + "token_train_data.pkl")
 
  training_sentences = train['tokens']
  testing_sentences = test['tokens']

  training_labels = train['labels']
  testing_labels = test['labels']

  return training_sentences, testing_sentences, training_labels, testing_labels

# tokenizing & padding 
def tokenize_and_pad_model3(path, vocab_size = 20000, embedding_dim = 100, max_length = 30, padding_type = 'post'):
  truct_type = 'post'
  oov_tok = '<OOV>'
  training_sentences, testing_sentences, training_labels, testing_labels = load_data(path)
  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
  tokenizer.fit_on_texts(training_sentences)
  # word_index = tokenizer.word_index

  training_sequences  = tokenizer.texts_to_sequences(training_sentences)
  training_padded = pad_sequences(training_sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=truct_type)

  testing_sequences  = tokenizer.texts_to_sequences(testing_sentences)
  testing_padded = pad_sequences(testing_sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=truct_type)
  training_padded = np.array(training_padded)
  training_labels = np.array(training_labels)

  testing_padded = np.array(testing_padded)
  testing_labels = np.array(testing_labels)

  return training_padded, training_labels, testing_padded, testing_labels

# modeling
def model3_context(path, dropout = 0.5, embedding_dim = 100, max_length=30, batch_size = 50, num_epochs = 10, l2_norm=0.003):
  filter_sizes = (3, 4, 5)
  num_filters = 100
  hidden_dims = 100
  vocab_size = 20000

  training_padded, training_labels, testing_padded, testing_labels = tokenize_and_pad_model3(path)

  conv_blocks =[]
  input_shape = (max_length, )
  model_input = tf.keras.layers.Input(shape=input_shape)
  z = model_input
  embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(z)
  for sz in filter_sizes:
      conv = tf.keras.layers.Conv1D(filters=num_filters,
                          kernel_size=sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(embedding)
      conv = tf.keras.layers.GlobalAveragePooling1D()(conv)
      conv = tf.keras.layers.Flatten()(conv)
      conv_blocks.append(conv)
  z = tf.keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

  z = tf.keras.layers.Dense(hidden_dims, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_norm), bias_regularizer=tf.keras.regularizers.l2(l2_norm))(z)
  z = tf.keras.layers.Dropout(dropout)(z)
  model_output = tf.keras.layers.Dense(1, activation="sigmoid")(z)

  model = tf.keras.Model(model_input, model_output)

  print(model.summary())

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  early_stopping=tf.keras.callbacks.EarlyStopping(patience=2)
  call_back = ready_callbacks(dir = 'ckpt3')
  history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), callbacks=call_back, batch_size = batch_size)


  return model, history

model, history = model3_context(path, num_epochs = 10)

# plot 출력 
def plot_accuracy_graphs(history, string='accuracy'):
  plt.plot(history.history[string])
  plt.plot(history.history['val_' + string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_' + string])
  plt.show()

def plot_loss_graphs(history, string='loss'):
  plt.plot(history.history[string])
  plt.plot(history.history['val_' + string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_' + string])
  plt.show()

plot_accuracy_graphs(history)
plot_loss_graphs(history)