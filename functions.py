import numpy as np
import pandas as pd
from konlpy.tag import Okt
from gensim import models
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot


def preprocessing(data):
    data.drop_duplicates(subset=['document'], inplace=True)
    data = data.dropna(how = 'any')
    data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
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
            if (i[1] == 'Noun' or i[1] == 'Adjective' or i[1] == 'Alpha'):                  
                result.append(i[0])
            
        tokenized_sentence.append(result)

    return tokenized_sentence

def pad_sequence(sentences, padding_word="<PAD/>"): #  오른쪽을 패딩주기
    maxlen = 40
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence)<=maxlen:
            num_padding = maxlen - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else : new_sentence = sentence[:maxlen]
        padded_sentences.append(new_sentence)
    return padded_sentences

def fasttext_vectorize(padded_sentences, max_len = 40):
    ko_model = models.fasttext.load_facebook_model('cc.ko.300.bin')
    paddedarray = np.array([ko_model.wv.word_vec(token) for x in padded_sentences for token in x])
    final_array = paddedarray.reshape(-1,max_len,300)
    return final_array

def simple_fasttext_vectorize(padded_sentences, max_len = 40):
    with open('simple_ko_vec.pkl','rb') as fw:
        simple_w2v= pickle.load(fw)
    ko_model = models.fasttext.load_facebook_model('cc.ko.300.bin')
    paddedarray=[]
    try:
        for x in padded_sentences:
            for token in x:
                paddedarray.append(simple_w2v[token])
        # paddedarray = np.array([simple_w2v[token] for x in padded_sentences for token in x])
    except:
        paddedarray.append(simple_w2v['얘쁜'])## 사전에 없는 단어는 0행렬 만듬 ['얘쁜']의 행렬이 0행렬이라 이렇게 그냥 썼음
    final_array = paddedarray.reshape(-1,max_len,300)
    return final_array

def plot_graphs(history, string, name='model'):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title(name)
    plt.legend([string, 'val_' + string])
    plt.show()
    ##저장될 폴더생성
    result_dir = './result_file'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(result_dir+'/{}.png'.format(name))
    print('<{}.png> result_file폴더에 결과 그래프 저장 완료'.format(name))


######################################
#######  model1:fastext이용 ###########
###################################### 
def model_1():
    # embedding_dim = 200
    filter_sizes = (3, 4, 5)
    num_filters = 100
    dropout = 0.5
    hidden_dims = 10
    batch_size = 50
    num_epochs = 10
    conv_blocks = []
    sequence_length = 200

# input_shape = (sequence_length, embedding_dim) # input shape for
    input_shape = (40, 300) # input shape for data, (max_length of sent, vect)

    model_input = keras.layers.Input(shape=input_shape)

    z = model_input
    for sz in filter_sizes:
        conv = keras.layers.Conv1D(filters=num_filters,
                            kernel_size=sz,
                            padding="valid",
                            activation="relu",
                            strides=1)(z)
        conv = keras.layers.MaxPooling1D(pool_size=2)(conv)
        conv = keras.layers.Flatten()(conv)
        conv_blocks.append(conv)
    z = keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = keras.layers.Dense(hidden_dims, activation="relu", kernel_regularizer=keras.regularizers.l2(0.003), bias_regularizer=keras.regularizers.l2(0.003))(z)
    z = keras.layers.Dropout(dropout)(z)
    model_output = keras.layers.Dense(1, activation="sigmoid")(z)

    model = keras.Model(model_input, model_output)

    # Create a new linear regression model.
    # model = keras.Sequential([keras.layers.Dense(1)])
    # model.compile(optimizer='adam', loss='mse')

    return model 


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
  for sz in filter_sizes:
      embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(z)
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



# model 3 실행법 
path = os.getcwd() #이런식으로 데이터 파일이 있는 path 지정 

model, history = model3_context(path) #모델실행 

#모델 그래프, 정확도 그래프, loss그래프 출력 
SVG(model_to_dot(model, show_layer_names=True, dpi=65).create(prog='dot', format='svg'))
plot_accuracy_graphs(history)
plot_loss_graphs(history)
    



