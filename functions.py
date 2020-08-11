import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import pickle

from konlpy.tag import Okt
from gensim import models
from gensim.models import Word2Vec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.vis_utils import model_to_dot

from IPython.display import SVG




max_len = 30

#데이터프레임을 받아서 한글이외의 값지우고 nan값 지우고 필요한 데이터만 반환 
def preprocessing(data):

    data.drop_duplicates(subset=['document'], inplace=True)
    data = data.dropna(how = 'any')
    data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣0_9A-Za-z ]","")

    data['document'].replace('', np.nan, inplace=True)
    data = data.dropna(how = 'any')
    sentences = data['document'].tolist()
    label = data['label']
    print('data len = {}'.format(len(sentences)))

    return sentences, label
    

def tokenize(sentence):
    
    okt = Okt()
    tokenized_sentence = []
    # 우선 단어의 기본형으로 모두 살리고, 명사, 동사, 영어만 담는다.
    # 그냥 nouns로 분리하는 것보다 좀 더 정확하고 많은 데이터를 얻을 수 있다.
    for line in sentence:
        result = []
        temp_sentence = okt.pos(line, norm=True, stem=True) # 먼저 형태소 분리해서 리스트에 담고

        for i in temp_sentence:                             
            if (i[1] == 'Noun' or i[1] == 'Adjective' or i[1] == 'Alpha' or i[1] == 'Verb' or i[1] == 'KoreanParticle' or i[1] == 'Number' ):                  
                result.append(i[0])
            
        tokenized_sentence.append(result)

    return tokenized_sentence

# 밑에꺼랑 같은데 영화리뷰 토큰화된거 사용하려고 만듬            
def input_preprocessing2(tokens):
    with open('tokenizer.pickle','rb') as f:
        tokenizer = pickle.load(f)

    sequences = tokenizer.texts_to_sequences(tokens)
    sequences_padded = pad_sequences(
        sequences, maxlen=30, padding='post', truncating='post')

    return sequences_padded


def input_preprocessing(sentences):
    tokens = tokenize(sentences)
    ######tokenizer 불러오기 vocab size = 20000으로 되어있는 거임
    with open('tokenizer.pickle','rb') as f:
        tokenizer = pickle.load(f)

    sequences = tokenizer.texts_to_sequences(tokens)
    sequences_padded = pad_sequences(
        sequences, maxlen=30, padding='post', truncating='post')

    return sequences_padded

def restore_model(model):
    if model=='model1':checkpoint_dir = './ckpt1'
    elif model=='model2': checkpoint_dir = './ckpt2'
    else : checkpoint_dir = './ckpt3'
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]

    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print('Restoring from', latest_checkpoint)
    return keras.models.load_model(latest_checkpoint)



def plot_graphs(history, string, name='model'):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title(name)
    plt.legend([string, 'val_' + string])

    fig = plt.gcf()
    ##저장될 폴더생성
    result_dir = './result_file'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig.savefig(result_dir+'/{}.png'.format(name), dpi = fig.dpi)
    print('<{}.png> result_file폴더에 결과 그래프 저장 완료'.format(name))
    plt.show()

def ready_callbacks(dir = './ckpt1'):
    import os #폴더 생성
    checkpoint_dir = dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=0),
        # This callback saves a SavedModel every 100 batches.
        # We include the training loss in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
            monitor='val_loss',
            save_best_only=True)
    ]
    return callbacks

def token_padded(sentences):
  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
  tokenizer.fit_on_texts(sentences)
  word_index = tokenizer.word_index
  sequences  = tokenizer.texts_to_sequences(sentences)
  padded = pad_sequences(sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=truct_type)
  return padded

def fasttext_vectorize(padded_sentences, max_len = 40):
    ko_model = models.fasttext.load_facebook_model('cc.ko.300.bin')
    paddedarray = np.array([ko_model.wv.word_vec(token) for x in padded_sentences for token in x])
    final_array = paddedarray.reshape(-1,max_len,300)
    return final_array

#tokenizer fit 해서 pkl 형태로 저장
def make_tokenizer_pkl():
    oov_tok = '<OOV>'
    truct_type = 'post'
    padding_type = 'post'
    max_length = 30
    vocab_size =20000
    training_sentences, training_labels, testing_sentences, testing_labels = m2_load_token_and_label()

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_idx = tokenizer.index_word

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('tokenizer 저장완료')
######################################
#######  model1:fastext이용 ###########
###################################### 
def CNN_model_1(dropout=0.5, num_filters=100, hidden_dims = 10, filter_sizes = (3, 4, 5), l2_norm = 0.003, max_len=30):
    
#저장되어있는 tokenizer 로드하기
    with open('tokenizer.pickle','rb') as f:
        tokenizer = pickle.load(f)

    word_index = tokenizer.word_index
#embedding_matrix 생성 fasttext기반 w2v
    vocab_size = len(word_index) + 1
    embedding_dim = 300

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open('simple_ko_vec.pkl','rb') as fw:
        ko_model= pickle.load(fw)
    for word, idx in tokenizer.word_index.items():
        embedding_vector = ko_model[word] if word in ko_model else None
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    filter_sizes = filter_sizes
    num_filters = num_filters
    dropout = dropout
    hidden_dims = hidden_dims
    l2_norm = l2_norm
    max_len = max_len

#model 만들기
    input_shape = (max_len) 
    model_input = keras.layers.Input(shape=input_shape)
    z = model_input
    embedding = keras.layers.Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],input_length=max_len,
                                      weights =[embedding_matrix], trainable = False)(z)
    conv_blocks = []
    for sz in filter_sizes:
        conv = keras.layers.Conv1D(filters=num_filters,
                            kernel_size=sz,
                            padding="valid",
                            activation="relu",
                            strides=1)(embedding)
        conv = keras.layers.MaxPooling1D(pool_size=2)(conv)
        conv = keras.layers.Flatten()(conv)
        conv_blocks.append(conv)
    z = keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = keras.layers.Dense(hidden_dims, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_norm), bias_regularizer=keras.regularizers.l2(l2_norm))(z)
    z = keras.layers.Dropout(dropout)(z)
    model_output = keras.layers.Dense(1, activation="sigmoid")(z)

    model = keras.Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Create a new linear regression model.
    # model = keras.Sequential([keras.layers.Dense(1)])
    # model.compile(optimizer='adam', loss='mse')

    return model 

#########################################
#######  model2:word2vec 이용 ###########
#########################################

''' 
  You need to check the path of these.

  test = pd.read_pickle("token_test_data.pkl")
  train = pd.read_pickle("token_train_data.pkl")
  ko_model= Word2Vec.load('word2vec_movie.model')

  plz add these files in the right folder.
'''

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


  # Sequence/ Padding
  training_sequences  = tokenizer.texts_to_sequences(training_sentences)
  training_padded = pad_sequences(training_sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=truct_type)


  testing_sequences  = tokenizer.texts_to_sequences(testing_sentences)
  testing_padded = pad_sequences(testing_sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=truct_type)
  # word2vec weight 
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
  callbacks = [
      keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=0),   
      keras.callbacks.ModelCheckpoint(
          filepath=checkpoint_dir + '/ckpt2-loss={loss:.3f}',
          save_freq=500)
      ]

  history = model.fit(training_padded, training_labels, epochs=10, callbacks=callbacks, batch_size = batch_size, validation_data=(testing_padded, testing_labels))
  accuracy_graph = plot_graphs(history, 'accuracy',name='model2_accuracy')
  loss_graph= plot_graphs(history, 'loss',name='model2_loss')  

  return model, history, accuracy_graph,loss_graph

# model2 실행및 그래프 생성
# model, history, accuracy_graph,loss_graph = m2_model()


######################################
###모델3_contextualized_embedding#####
######################################
def load_data(path):
    test = pd.read_pickle(path +'/'+ "token_test_data.pkl")
    train = pd.read_pickle(path +'/' + "token_train_data.pkl")
    
    training_sentences = train['tokens']
    testing_sentences = test['tokens']

    training_labels = train['labels']
    testing_labels = test['labels']

    return training_sentences, testing_sentences, training_labels, testing_labels

def tokenize_and_pad_model3(path, vocab_size = 20000, embedding_dim = 100, max_length = 30, padding_type = 'post'):
  truct_type = 'post'
  oov_tok = '<OOV>'
  training_sentences, testing_sentences, training_labels, testing_labels = load_data(path)
  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
  tokenizer.fit_on_texts(training_sentences)
  word_index = tokenizer.word_index

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
  history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), callbacks=[early_stopping], batch_size = batch_size)

  return model, history

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



# # model 3 실행법 
# path = os.getcwd() #이런식으로 데이터 파일이 있는 path 지정 

# model, history = model3_context(path) #모델실행 

# #모델 그래프, 정확도 그래프, loss그래프 출력 
# SVG(model_to_dot(model, show_layer_names=True, dpi=65).create(prog='dot', format='svg'))
# plot_accuracy_graphs(history)
# plot_loss_graphs(history)
    

