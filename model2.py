
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
from gensim.models import Word2Vec
import os
from tensorflow import keras
from functions import ready_callbacks

''' 
  You need to check the path of these.

  test = pd.read_pickle("token_test_data.pkl")
  train = pd.read_pickle("token_train_data.pkl")
  ko_model= Word2Vec.load('word2vec_movie.model')

  plz add these files in the right folder.
'''

def naver_w2v():
  df_train = pd.read_pickle('token_train_data.pkl')
  df_test = pd.read_pickle('token_test_data.pkl')
  df = pd.concat([df_train, df_test], axis = 0, ignore_index=True)

  token = [i  for x in df['tokens'] for i in x]

  model = Word2Vec(sentences =token , size = 200, window = 5, min_count = 5, workers = 4, sg = 0)
  model.save('word2vec_movie.model')


# made by ChangYoon
def plot_graphs(history, string, name='model'):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title(name)
    plt.legend([string, 'val_' + string])
    
    ##저장될 폴더생성
    result_dir = './result_file'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(result_dir+'/{}.png'.format(name))
    print('<{}.png> result_file폴더에 결과 그래프 저장 완료'.format(name))
    plt.show()

def m2_load_token_and_label():

  test = pd.read_pickle("token_test_data.pkl")
  train = pd.read_pickle("token_train_data.pkl")

  training_sentences, training_labels = train['tokens'], train['labels']
  testing_sentences, testing_labels = test['tokens'], test['labels']

  return training_sentences, training_labels, testing_sentences, testing_labels


def m2_tokenizer():

  vocab_size = 20000
  embedding_dim = 200
  max_length = 30
  truct_type = 'post'
  padding_type = 'post'
  oov_tok = '<OOV>'

  training_sentences, training_labels, testing_sentences, testing_labels = m2_load_token_and_label()

  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
  tokenizer.fit_on_texts(training_sentences)
  word_idx = tokenizer.index_word


  # Sequence, Padding
  training_sequences  = tokenizer.texts_to_sequences(training_sentences)
  training_padded = pad_sequences(training_sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=truct_type)


  testing_sequences  = tokenizer.texts_to_sequences(testing_sentences)
  testing_padded = pad_sequences(testing_sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=truct_type)
  #word2vec weight
  vocab_size = len(word_idx) + 1
  embedding_dim = 200

  embedding_matrix = np.zeros((vocab_size, embedding_dim))
  ko_model= Word2Vec.load('word2vec_movie.model')

  for word, idx in tokenizer.word_index.items():
      embedding_vector = ko_model[word] if word in ko_model else None
      if embedding_vector is not None:
          embedding_matrix[idx] = embedding_vector

  return training_padded, testing_padded, training_labels,testing_labels,embedding_matrix, vocab_size

def m2_model():

  embedding_dim = 200
  filter_sizes = (3, 4, 5)
  num_filters = 100
  dropout = 0.5
  hidden_dims = 100
  max_length = 30

  conv_blocks =[]
  input_shape = (30)
  model_input = tf.keras.layers.Input(shape=input_shape)
  z = model_input

  training_padded, testing_padded, training_labels,testing_labels,embedding_matrix, vocab_size = m2_tokenizer()
  
  embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length,
                                          weights = [embedding_matrix], trainable = False)(z)  
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

  z = tf.keras.layers.Dense(hidden_dims, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.003), bias_regularizer=tf.keras.regularizers.l2(0.003))(z)
  z = tf.keras.layers.Dropout(dropout)(z)
  model_output = tf.keras.layers.Dense(1, activation="sigmoid")(z)
  model = tf.keras.Model(model_input, model_output)

  batch_size = 50
  num_epochs = 10
  min_word_count = 1
  context = 10

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  checkpoint_dir = './ckpt2'
  if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
  callbacks = ready_callbacks('ckpt2')

  history = model.fit(training_padded, training_labels, epochs=10, callbacks=callbacks, batch_size = batch_size, validation_data=(testing_padded, testing_labels))
  accuracy_graph = plot_graphs(history, 'accuracy',name='model2_accuracy')
  loss_graph= plot_graphs(history, 'loss',name='model2_loss')  

  return model, history, accuracy_graph,loss_graph



model, history, accuracy_graph,loss_graph = m2_model() # Model2 실행 및 그래프 작성
