# Convolution Neural Networks for Sentence Classification 
<img align="right" src="https://img.shields.io/badge/python-3.6-green"><img align="right" src="https://img.shields.io/badge/tensorflow-2.3-green">

#### contributers  ![name](https://img.shields.io/badge/-김민균-black) ![https://img.shields.io/badge/-%EC%9D%B4%EC%86%8C%EC%97%B0-black](https://img.shields.io/badge/-이소연-black) ![name](https://img.shields.io/badge/-이창윤-black) ![name](https://img.shields.io/badge/-나서연-black)

CNN모델을 기반으로 다양한 embedding 방식을 이용하여 결과를 비교해 본 프로젝트 입니다.



### Reaquirements

- festtext사이트에서 한글 wordvector 다운로드 (cc.ko.300.bin)

- tensorflow



### Usage

model 학습 :  model1.py / model2.py/ model3.py 실행 하여 학습 진행하면 각각의 이름의 폴더에 model이 저장된다.

model test : test.py 에 각각의 모델을 선택하여 학습된 모델을 불러와 임의로 생성한 리뷰문서에대한 평가를 해 볼 수 있다

evaluate.py를 이용하여 blue score를 계산해 볼수 있다

```
$ python evaluate.py --model 학습된MODEL --tokenizer data/tokenizer.pickle --param data/parameter.json --dataset data/review.tsv
```





# 개요

![model3_model.png](https://github.com/hamakim94/CNN_Project/blob/master/result_file/model3_model.png?raw=true)

#### model1 : Static / using fasttext, 

num_filters=100, hidden_dims = 10, filter_sizes = (3, 4, 5), l2_norm = 0.003

dropout = 0.5, max_length = 53, input_shape = (53, 300)

#### model 2 : Non-Static / using Word2Vec(C)

embedding_dim = 200  filter_sizes = (3, 4, 5)  num_filters = 100
 dropout = 0.5  hidden_dims = 100  max_length = 30  input_shape = (30)

#### model3 : Contextualized Embedding 

dropout = 0.5, embedding_dim = 100, max_length=30, batch_size = 50, num_epochs = 10 ):
  filter_sizes = (3, 4, 5),  num_filters = 100,  hidden_dims = 100,  vocab_size = 20000



## 공통적으로 처리한 일(전처리)

#### 1. 중복된 데이터 개수 확인

```python
train_data['document'].nunique(), train_data['label'].nunique() 
test_data['document'].nunique(), test_data['label'].nunique() 
```

#### 2. 중복된 데이터 삭제

```python
train_data.drop_duplicates(subset=['document'], inplace=True) 
test_data.drop_duplicates(subset=['document'], inplace=True)
```

#### 3. null값 찾기, 지워주기

```python
print(train_data.isnull().values.any()) # 이게 true가 나오면 결측치 있음
 -> True
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
test_data = test_data.dropna(how = 'any') #
print(train_data.isnull().values.any(),test_data.isnull().values.any()) 
  -> False False
```

#### 4. 영어 한글 자음 모음만 뽑기, 빈 칸 제거

```python
train_data['document'] = train_data['document'].str.replace("[^a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_data['document'] = test_data['document'].str.replace("[^a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]","")

train_data['document'].replace('', np.nan, inplace=True)  # 빈칸 -> 한글
test_data['document'].replace('', np.nan, inplace=True) 
```

### 함수로

```python
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
```

## 네이버 영화(한글) 데이터를 Konlpy.tag 에 Okt를 활용하여 Noun, Adjective, Alpha(영어)만 뽑음

konlpy의 okt를 이용해 포스 태깅, 명사, 형용사, 영어만 뽑음

```python
def tokenize(sentence):
    okt = Okt()
    tokenized_sentence = []
for line in sentence:
    result = []
    temp_sentence = okt.pos(line, norm=True, stem=True)
    print(temp_sentence)
    for i in temp_sentence:                             
        if (i[1] == 'Noun' or i[1] == 'Adjective' or i[1] == 'Alpha'):                  
            result.append(i[0])
        
    tokenized_sentence.append(result)

return tokenized_sentence
```
토큰화된 문장을 pkl 파일로 만들어 사용하기 편하게 했음

cf. 문서당 뽑힌 토큰의 개수 히스토그램



![네이버영화리뷰토큰길이hist.PNG](https://github.com/hamakim94/CNN_Project/blob/master/result_file/%EB%84%A4%EC%9D%B4%EB%B2%84%EC%98%81%ED%99%94%EB%A6%AC%EB%B7%B0%ED%86%A0%ED%81%B0%EA%B8%B8%EC%9D%B4hist.PNG?raw=true)

max_length의 기준을 30(model1, 2,3)으로 설정하기로 했음



## 토크나이저를 pkl로 만들어주는 함수

```python
#tokenizer fit 해서 pkl 형태로 저장
def make_tokenizer_pkl():
    oov_tok = '<OOV>'
    truct_type = 'post'
    padding_type = 'post'
    max_length = 30
    vocab_size =20000
    training_sentences, training_labels, testing_sentences, testing_labels = m2_load_token_and_label()
	# m2_load_token_and_label() -> (전처리, 라벨링 된 pkl 불러오는 함수)
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_idx = tokenizer.index_word

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('tokenizer 저장완료')
```




--------------------

## 1.  Static (using FastText), 

```python
def fasttext_vectorize(padded_sentences, max_len = 40):
    ko_model = models.fasttext.load_facebook_model('cc.ko.300.bin')
    paddedarray = np.array([ko_model.wv.word_vec(token) for x in padded_sentences for token in x])
    final_array = paddedarray.reshape(-1,max_len,300)
    return final_array
```



**해당 작업은 직접 fasttext bin을 불러들이기때문에, 오래걸려서 따로 벡터화해서 만들었음**



```python
simple_ko_vec = {}
oov_list=[]
simple_ko_vec['<PAD/>'] = ko_model.wv.word_vec('<PAD/>')
for sent in token_train_data+token_test_data:
    for token in sent:
        try:
            simple_ko_vec[token] = ko_model.wv.word_vec(token)
        except:
            oov_list.append(token)
oov_list= list(set(oov_list))
print(len(oov_list))
print(len(simple_ko_vec))

with open('simple_ko_vec.pkl','wb') as fw:
    pickle.dump(simple_ko_vec, fw)
```

**기존 토큰들로 dictionary 만들어 simple_ko_vec.pkl로 저장해서, 불러들임**

```python
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
    for word, idx in tokenizer.word_index.items(): # word_index에 있는건만 FastText Vector 할당
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
    embedding = keras.layers.Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],
                                       input_length=max_len, weights =[embedding_matrix], 
                                       trainable = False)(z)
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

    return model 
```

**callback설정 함수 : 1epoch가 지날떄마다 checkpoint에 모델을 저장한다,  val_loss가 낮아지는 시점에 그만한다**

```python
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
```

**model fit 시키고 plot 저장한다**

```python
    model = CNN_model_1()     
    callbacks = ready_callbacks('ckpt1')

    # 모델 fit 시키기
    history = model.fit(X_train, y_train, epochs=num_epochs, callbacks=callbacks, 
                        batch_size = batch_size, validation_data=(X_test, y_test))

    #CNN 모델에 fit한 history, string: ('accuracy'or'loss'), name : 해당이름으로 차트title과 차트파일명이 됨
    #result_file폴더에 결과 그래프 저장
    fc.plot_graphs(history, 'accuracy',name='model1_acc')
    fc.plot_graphs(history, 'loss', name='model1_loss')
```



## 결과

```
Epoch 1/10
2915/2916 [============================>.] - ETA: 0s - loss: 0.5174 - accuracy: 0.7585WARNING:tensorflow:From C:\Users\User\anaconda3\envs\nlp\lib\site-packages\tensorflow\python\training\tracking\tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
WARNING:tensorflow:From C:\Users\User\anaconda3\envs\nlp\lib\site-packages\tensorflow\python\training\tracking\tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
INFO:tensorflow:Assets written to: ./ckpt1\ckpt-loss=0.517\assets
2916/2916 [==============================] - 148s 51ms/step - loss: 0.5175 - accuracy: 0.7585 - val_loss: 0.4468 - val_accuracy: 0.8034
Epoch 2/10
2915/2916 [============================>.] - ETA: 0s - loss: 0.4550 - accuracy: 0.8121INFO:tensorflow:Assets written to: ./ckpt1\ckpt-loss=0.455\assets
2916/2916 [==============================] - 147s 51ms/step - loss: 0.4550 - accuracy: 0.8121 - val_loss: 0.4349 - val_accuracy: 0.8129
Epoch 3/10
2916/2916 [==============================] - 135s 46ms/step - loss: 0.4221 - accuracy: 0.8333 - val_loss: 0.4357 - val_accuracy: 0.8147
Epoch 4/10
2916/2916 [==============================] - 126s 43ms/step - loss: 0.3918 - accuracy: 0.8555 - val_loss: 0.4509 - val_accuracy: 0.8128
<model1_acc.png> result_file폴더에 결과 그래프 저장 완료
```
```
<model1_loss.png> result_file폴더에 결과 그래프 저장 완료
```

<p align="center">
  <img width="460" height="350" src="https://github.com/hamakim94/CNN_Project/blob/master/result_file/model1_acc.png?raw=true"><img width="460" height="350" src="https://github.com/hamakim94/CNN_Project/blob/master/result_file/model1_loss.png?raw=true">
</p>









## 2. Non_Static (using Word2Vec - naver train data)

#### 앞서 전처리된 train, test 데이터를 불러옴

```python
df_train = pd.read_pickle('token_train_data.pkl')
df_test = pd.read_pickle('token_test_data.pkl')

# train test 합쳐서 사용
df = pd.concat([df_train, df_test], axis = 0, ignore_index=True)
token = [i  for x in df['tokens'] for i in x]
```

#### gensim의 Word2Vec을 이용하여 Vectorizing 한다.

```python
# word2vec  모델생성
model = Word2Vec(sentences =token , size = 200, window = 5, min_count = 5, workers = 4, sg = 0)  # 200차원, window size 5로 설정, min_count = 5로
# sg = 0은 CBOW, 1은 Skip-gram. CBOW 
# 모델 저장
model.save('new_word2vec_movie.model')
```

#### 

#### 데이터를 불러들인다.

```python
def m2_load_token_and_label():

  test = pd.read_pickle("token_test_data.pkl")
  train = pd.read_pickle("token_train_data.pkl")

  training_sentences, training_labels = train['tokens'], train['labels']
  testing_sentences, testing_labels = test['tokens'], test['labels']

  return training_sentences, training_labels, testing_sentences, testing_labels

```

### token을 만들어 낸 후 padding 한다. 

```python
def m2_tokenizer():

  vocab_size = 20000
  embedding_dim = 200
  max_length = 30
  truct_type = 'post' #( 뒤쪽 )
  padding_type = 'post'
  oov_tok = '<OOV>'

  training_sentences, training_labels, testing_sentences, testing_labels = m2_load_token_and_label()

  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
  tokenizer.fit_on_texts(training_sentences)
  word_idx = tokenizer.index_word


  # Sequence 만들기 / Padding하기
  training_sequences  = tokenizer.texts_to_sequences(training_sentences)
  training_padded = pad_sequences(training_sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=truct_type)


  testing_sequences  = tokenizer.texts_to_sequences(testing_sentences)
  testing_padded = pad_sequences(testing_sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=truct_type)
  # word2vec weight 
  # tokenizer에 있는 단어 사전을 순회하면서 word2vec의 200차원 vector를 가져오기
  vocab_size = len(word_idx) + 1 ## oov 때문
  embedding_dim = 200

  embedding_matrix = np.zeros((vocab_size, embedding_dim))
  ko_model= Word2Vec.load('new_word2vec_movie.model')

  for word, idx in tokenizer.word_index.items():
      embedding_vector = ko_model[word] if word in ko_model else None
      if embedding_vector is not None:
          embedding_matrix[idx] = embedding_vector

  return training_padded, testing_padded, training_labels,testing_labels,embedding_matrix, vocab_size
```

```python
vocab_size # 총 사전의 길이
```

```python
40015
```

tokenize 된 단어의 갯수만큼의 크기를 가진 embedding matrix를 생성한다.

```python
embedding_matrix.shape
```

```python
(40015, 200)
```



### 모델링 함수

```python
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

  embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length, weights = [embedding_matrix], trainable = False)(z)
  
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

# compile용 변수들
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
```

### 모델 컴파일 및 훈련

```python
model, history, accuracy_graph,loss_graph = m2_model()
```

```python
Epoch 1/10
[==============================] - 145s 50ms/step - loss: 0.4559 - accuracy: 0.8007 - val_loss: 0.4218 - val_accuracy: 0.8075
Epoch 2/10
[==============================] - 148s 51ms/step - loss: 0.4208 - accuracy: 0.8115 - val_loss: 0.4117 - val_accuracy: 0.8137
Epoch 3/10
[==============================] - 149s 51ms/step - loss: 0.4103 - accuracy: 0.8184 - val_loss: 0.4057 - val_accuracy: 0.8165
Epoch 4/10
[==============================] - 147s 51ms/step - loss: 0.4032 - accuracy: 0.8206 - val_loss: 0.4053 - val_accuracy: 0.8192
Epoch 5/10
[==============================] - 150s 51ms/step - loss: 0.3968 - accuracy: 0.8258 - val_loss: 0.4035 - val_accuracy: 0.8209
Epoch 6/10
[==============================] - 146s 50ms/step - loss: 0.3906 - accuracy: 0.8287 - val_loss: 0.4046 - val_accuracy: 0.8195
                
```

<p align="center">
  <img width="459" height="279" src="https://github.com/hamakim94/CNN_Project/blob/master/result_file/model2_accuracy.png?raw=true"><img width="459" height="279" src="https://github.com/hamakim94/CNN_Project/blob/master/result_file/model2_loss.png?raw=true">
</p>








## 3. Contextualized Embedding

#### 데이터 불러들여, 원하는 변수들 만들기

```python
test = pd.read_pickle("token_test_data.pkl")
train = pd.read_pickle("token_train_data.pkl")
training_sentences = train['tokens']
testing_sentences = test['tokens']

training_labels = train['labels']
testing_labels = test['labels']
```

#### 파라미터 명시

```python
vocab_size = 20000      (20000개의 단어들만 쓸래)
embedding_dim = 200     ( 200차원의 벡터로 )
max_length = 30         (문장 최대 길이는 53으로)
truct_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
```

#### Tokenize

```python
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
```

#### Sequence 만들고, 패딩 

```python
training_sequences  = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, 
                                padding=padding_type, truncating=truct_type)


testing_sequences  = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, 
                                padding=padding_type, truncating=truct_type)
```

#### array로 바꾸기

```python
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)

testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
```

```
training_padded.shape
```

```
(145791, 53)
```

#### 모델링

```python
def model3_context(path, dropout = 0.5, embedding_dim = 100, max_length=30, batch_size = 50, num_epochs = 10 ):
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

  z = tf.keras.layers.Dense(hidden_dims, activation="relu")(z)
  z = tf.keras.layers.Dropout(dropout)(z)
  model_output = tf.keras.layers.Dense(1, activation="sigmoid")(z)

  model = tf.keras.Model(model_input, model_output)

  print(model.summary())

  batch_size = 50
  min_word_count = 1
  context = 10

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  early_stopping=tf.keras.callbacks.EarlyStopping(patience=2)
  call_back = ready_callbacks(dir = 'ckpt3')
  history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), callbacks=call_back, batch_size = batch_size)


  return model, history
```

#### 모델 컴파일, 훈련

```python
batch_size = 50
num_epochs = 10
min_word_count = 1
context = 10

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

```python
# model training 
num_epochs = 30
early_stopping=tf.keras.callbacks.EarlyStopping(patience=2)
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), callbacks=[early_stopping], batch_size = batch_size)

Epoch 1/30
2916/2916 [==============================] - 286s 98ms/step - loss: 0.4389 - accuracy: 0.8036 - val_loss: 0.3900 - val_accuracy: 0.8288
Epoch 2/30
2916/2916 [==============================] - 273s 94ms/step - loss: 0.3524 - accuracy: 0.8508 - val_loss: 0.3894 - val_accuracy: 0.8303
Epoch 3/30
2916/2916 [==============================] - 274s 94ms/step - loss: 0.3065 - accuracy: 0.8755 - val_loss: 0.4247 - val_accuracy: 0.8303
Epoch 4/30
2916/2916 [==============================] - 271s 93ms/step - loss: 0.2577 - accuracy: 0.8986 - val_loss: 0.4323 - val_accuracy: 0.8191
```

![model3_accuracy.png](https://github.com/hamakim94/CNN_Project/blob/master/result_file/model3_accuracy.png?raw=true)![model3_loss.png](https://github.com/hamakim94/CNN_Project/blob/master/result_file/model3_loss.png?raw=true)

epoch가 커질수록 train_data에 대한 accuracy가 높아지지만, overfit 된다



## 해당 모델들을 불러들이는 함수

```python
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
```

(위 함수가 아래 link py파일에 쓰임)

특정 모델을 불러들여서, 전처리를 해주고 예측해주는 함수

https://github.com/hamakim94/CNN_Project/commit/7aae83d3f1bbac7d157273b4ee1a708c3b5c89a1

만든 functions.py

https://github.com/hamakim94/CNN_Project/blob/master/functions.py

## TEST(model3예시)

```python
doc = [
    '좋은게 하나도 없네',
    '돈아깝다 개같은거',
    '깔게없는영화',
    '아직 못 본 사람 부럽다. 소름 돋을 수 있어서...처음 봤을 때 그 전율을 다시 한번 느끼고싶다.',
    '아주 재미지네요 하하하ㅏ',
    '완전 헛소리하고 있네',
    '오늘 밥을 먹었다',
    '관객에게 놀란이 실행하는 인셉션',
    '세번째 다시 봅니다',
    '어쭙잖은 부성애 vs 명분없는 형제애',
    '박정민 진짜 열일한다',
    '마틸다 기출변형 의상을 앞에 두고 감독은 설득한다. 이 영화의 휴머니즘은 당신의 천진난만한\
     뱃살에서 나온다고, 물색없이 아메리카노만 마시며',
    '개노잼인데 이정재가 송민호처럼 입고다닌는 것만 좀웃기다',
    '세 사람의 테이큰',
    '학생증 검사해서 고사 못보게 하고 이 영화를 보게 해주신 영화관 직원분 감사합니다.',
    '이걸 영화관에서 안봤다는게 내 인생 최악의 실수다'

]
y= [0,0,1,1,1,0,0,1,1,0,1,1,0,1,1,1]
X = input_preprocessing(doc)
print(X)
model = restore_model('model3') #model1|model2|model3 입력하면 각각의 저장된모델중 가장 최신의것을 골라서 불러온다
y = np.array(y)
print(model.summary())

yhat = model.predict(X)
        # Compute the loss value for this batch.
for i in range(len(yhat)):
    print('실제값:{}, 예측값:{}'.format(y[i], yhat[i]))
result = model.evaluate(X,y)
print('정답률은 {}% 입니다.'.format(100*result[1]))
```

> 실제값:0, 예측값:[0.30392236]\
> 실제값:0, 예측값:[0.0650005]\
> 실제값:1, 예측값:[0.21971333]\
> 실제값:1, 예측값:[0.99392986]\
> 실제값:1, 예측값:[0.6215658]\
> 실제값:0, 예측값:[0.83809507]\
> 실제값:0, 예측값:[0.22184402]\
> 실제값:1, 예측값:[0.68833643]\
> 실제값:1, 예측값:[0.8921679]\
> 실제값:0, 예측값:[0.39271945]\
> 실제값:1, 예측값:[0.78049505]\
> 실제값:1, 예측값:[0.4866558]\
> 실제값:0, 예측값:[0.08552262]\
> 실제값:1, 예측값:[0.51187897]\
> 실제값:1, 예측값:[0.45225626]\
> 실제값:1, 예측값:[0.03050807]\
> 1/1 [==============================] - 0s 0s/step - loss: 0.7262 - accuracy: 0.6875
> 정답률은 68.75% 입니다.

---



**test 값은 model1 : 40%대, model2 : 50%대로 model3가 제일 높았다**



## 결론

#### val_accuracy

| model1                    | model2                 | model3                  |
| ------------------------- | ---------------------- | ----------------------- |
| 0.8147                    | 0.8209                 | 0.8303                  |
| Total params : 12,404,821 | Total params:8,273,501 | Total params: 2,150,501 |

1. .py, functions.py 만드는데 어려움이 많았음
2. 실제로 sentence_embedding 방법이 가장 val_accuracy가 높았고, 최신 기술 순으로 높아짐
3. Stopwords 처리를 못했고, pos_tag의 종류들이 적음, 더 시도해보면 좋은 결과값이 나올것이라 예상
4. Total params의 갯수도 최신 기술 순으로 적어서 효율 적이다

