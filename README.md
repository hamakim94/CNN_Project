Convolution Neural Networks for Sentence Classification 

김민균,이소연,이창윤,나서연



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

## 네이버 영화(한글) 데이터를 Konlpy.tag 에 Okt를 활용하여 Noun, Adjective, Alpha(영어), Verb만 뽑음

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
        if (i[1] == 'Noun' or i[1] == 'Adjective' or i[1] == 'Alpha' or i[1] == 'Verb'):                  
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



**해당 작업은 직접 bin을 불러들이기때문에, 오래걸려서 따로 벡터화해서 만들었음**



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

![img](data:image/png;base64,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)

```
<model1_loss.png> result_file폴더에 결과 그래프 저장 완료
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYgAAAEWCAYAAAB8LwAVAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3dd3hUZdrH8e89k0Yg9EAghdBFipQQiQWxoK4F1gqCIIiCuupaX3Wrq1t1LWsXFuwKrGXXFVdEEQGlJPQmIYQWauglpt/vH3MCQxwghAwnk9yf68qVzCkz93FwfvOc5zzPEVXFGGOMKc/jdgHGGGOqJwsIY4wxAVlAGGOMCcgCwhhjTEAWEMYYYwKygDDGGBOQBYQxxpiALCCMOQki8qaI/LGC264XkUtO8fVO+TmMqSwLCGNOAxHpIiJTRWSniNjoVBMSLCCMOT2KgMnAKLcLMaaiLCBMjeScmnlYRJaKyCERGS8izUXkfyJyQES+EpFGzrYDRGSFiOwVkRki0snveXqIyEJnn0lAVLnXuUpEFjv7fi8i3QLVo6qrVXU8sOIUjilSRJ4XkS3Oz/MiEumsayoinzl17BaRWSLicdY9IiKbnWNYLSIXV7YGU7tYQJia7DqgP9ABuBr4H/AroCm+f/v3ikgH4APgPiAW+Bz4r4hEiEgE8G/gHaAx8C/nOQEQkZ7ABGAM0AR4Hfi07EM7CH4N9AG6A2cBqcBvnHUPAjnOMTR3jlNFpCNwN9BbVWOAy4D1QarP1DAWEKYme1FVt6vqZmAWME9VF6lqAfAJ0AMYBExR1WmqWgT8HagDnIPvwzgceF5Vi1T1QyDd7/lvB15X1XmqWqKqbwEFzn7BMBR4QlV3qGou8AdgmLOuCGgBtHJqnaW+mThLgEjgTBEJV9X1qro2SPWZGsYCwtRk2/3+/jHA43pAS2BD2UJVLQU2AfHOus169JTHG/z+bgU86JzW2Ssie4FEZ79gOKpW5++y13oayAK+FJFsEXkUQFWz8LWOHgd2iMhEEQlWfaaGsYAwtd0WfB/0AIiI4PuQ3wxsBeKdZWWS/P7eBPxJVRv6/USr6geno1anli0AqnpAVR9U1Tb4Tqc9UNbXoKrvq+p5zr4K/C1I9ZkaxgLC1HaTgStF5GIRCcd3Lr8A+B6YAxTj66sIE5Fr8Z33LzMOuENEzhafuiJypYjElH8RZ30UEOE8jqpEX8UHwG9EJFZEmgK/A951nu8qEWnnhNl+fKeWSkSko4hc5LxWPr6WU8lJvq6ppSwgTK2mqquBm4EXgZ34vn1fraqFqloIXAuMAPbg66/42G/fDHz9EC8567OcbQNphe/Duewqph+B1SdZ7h+BDGApsAxY6CwDaA98BRzEF2yvqOoMfP0Pf3WObRvQDF8HtjEnJHZHOWOMMYFYC8IYY0xAYW4XYExtJiJJwMpjrD5TVTeeznqM8WenmIwxxgRUY1oQTZs21eTkZLfLMMaYkLJgwYKdqhobaF2NCYjk5GQyMjLcLsMYY0KKiGw41jrrpDbGGBNQUANCRC53Zo/MKhv6X279CBHJdWbDXCwitznLu4vIHGeGzaUiMiiYdRpjjPmpoJ1iEhEv8DK+2TRzgHQR+VRVy1+xMUlV7y63LA8YrqprnHljFojIVFXdG6x6jTHGHC2YfRCpQJaqZgOIyERgIMe+pO8wVc30+3uLiOzAN42xBYQx5ihFRUXk5OSQn5/vdinVWlRUFAkJCYSHh1d4n2AGRDy+yczK5ABnB9juOhHpC2QC96uq/z6ISCq++Wt+MkWxiIwGRgMkJSWVX22MqQVycnKIiYkhOTmZo+dVNGVUlV27dpGTk0Pr1q0rvF8w+yACvVPlB138F0hW1W745pF566gnEGmB72YtI51pmI9+MtWxqpqiqimxsQGv0jLG1HD5+fk0adLEwuE4RIQmTZqcdCsrmAGRg2/a5DIJOFMTl1HVXc7NW8A3M2avsnUiUh+YAvxGVecGsU5jTIizcDixyvw3CmZApAPtRaS1c+vGwcCn/hs4LYQyA4BVzvIIfHf8eltV/xXEGikpVf78+Spy9uQF82WMMSbkBC0gVLUY371wp+L74J+sqitE5AkRGeBsdq9zKesS4F6OTJV8I9AXGOF3CWz3YNS5cXceE+dvZNDrc9mw61AwXsIYU8PVq1fP7RKCosbMxZSSkqKVHUm9fPM+ho2fR0SYh/du60O7ZjXzzTamJlq1ahWdOnVytYZ69epx8OBBV2uoiED/rURkgaqmBNreRlIDXeIbMHF0GiWlyuCxc1i97YDbJRljQpCq8vDDD9OlSxe6du3KpEmTANi6dSt9+/ale/fudOnShVmzZlFSUsKIESMOb/vcc8+5XP1P1Zi5mE5Vx7gYJo5OY+g/5zJ47BzeGXU2XeIbuF2WMeYk/OG/K1i5ZX+VPueZLevz+6s7V2jbjz/+mMWLF7NkyRJ27txJ79696du3L++//z6XXXYZv/71rykpKSEvL4/FixezefNmli9fDsDevdVvmJe1IPy0a1aPyWPSiI4IY8i4uSzauMftkowxIWT27NncdNNNeL1emjdvzgUXXEB6ejq9e/fmjTfe4PHHH2fZsmXExMTQpk0bsrOzueeee/jiiy+oX7++2+X/hLUgymnVpC6TxvRhyLh5DBs/nzdG9qZ3cmO3yzLGVEBFv+kHy7H6dPv27cvMmTOZMmUKw4YN4+GHH2b48OEsWbKEqVOn8vLLLzN58mQmTJhwmis+PmtBBJDQKJrJY9JoVj+S4ePn833WTrdLMsaEgL59+zJp0iRKSkrIzc1l5syZpKamsmHDBpo1a8btt9/OqFGjWLhwITt37qS0tJTrrruOJ598koULF7pd/k9YC+IY4hpEMWl0Gjf/cx4j30zn9WG96NexmdtlGWOqsWuuuYY5c+Zw1llnISI89dRTxMXF8dZbb/H0008THh5OvXr1ePvtt9m8eTMjR46ktNQ3ScRf/vIXl6v/KbvM9QR2Hypk2Ph5rNl+kJeG9ODSznFV/hrGmMqrDpe5hgq7zLWKNa4bwfu39eHMlvW5672FTFm61e2SjDHmtLCAqIAG0eG8MyqVHkkNueeDhXyyKMftkowxJugsICooJiqct25NpU+bJjwweQkT5290uyRjjAkqC4iTEB0RxoQRvenbPpZHP17G23PWu12SMcYEjQXESYoK9zJ2eC/6n9mc3/1nBeNmZrtdkjHGBIUFRCVEhnl5ZWhPruzagj99voqXpq9xuyRjjKlyNg6iksK9Hv4xuDuRYR7+/mUmBcWlPNC/g924xBhTY1gL4hSEeT08fcNZDO6dyIvTs/jL/3445lB7Y4yB4987Yv369XTp0uU0VnN81oI4RV6P8OdruhIR5mHszGwKikr4/dWd8XisJWGMCW0WEFXA4xH+MKAzkWEexs1aR0FxKX++pquFhDGn2/8ehW3LqvY547rCz/56zNWPPPIIrVq14q677gLg8ccfR0SYOXMme/bsoaioiD/+8Y8MHDjwpF42Pz+fO++8k4yMDMLCwnj22We58MILWbFiBSNHjqSwsJDS0lI++ugjWrZsyY033khOTg4lJSX89re/ZdCgQad02GABUWVEhF9d0YmocC8vTs+isLiUp67vRpjXzuIZU5MNHjyY++6773BATJ48mS+++IL777+f+vXrs3PnTvr06cOAAQNOqo/y5ZdfBmDZsmX88MMPXHrppWRmZvLaa6/xy1/+kqFDh1JYWEhJSQmff/45LVu2ZMqUKQDs27evSo7NAqIKiQgPXtqRCK+HZ6b5Oq6fH9ydcAsJY06P43zTD5YePXqwY8cOtmzZQm5uLo0aNaJFixbcf//9zJw5E4/Hw+bNm9m+fTtxcRWfy2327Nncc889AJxxxhm0atWKzMxM0tLS+NOf/kROTg7XXnst7du3p2vXrjz00EM88sgjXHXVVZx//vlVcmz2yRUE91zcnl9f0Ykpy7Zy13sLKSgucbskY0wQXX/99Xz44YdMmjSJwYMH895775Gbm8uCBQtYvHgxzZs3Jz8//6Se81gXvAwZMoRPP/2UOnXqcNlllzF9+nQ6dOjAggUL6Nq1K4899hhPPPFEVRyWBUSw3N63DU8M7My0ldsZ/fYC8ossJIypqQYPHszEiRP58MMPuf7669m3bx/NmjUjPDycb775hg0bNpz0c/bt25f33nsPgMzMTDZu3EjHjh3Jzs6mTZs23HvvvQwYMIClS5eyZcsWoqOjufnmm3nooYeq7N4SdoopiIanJRPh9fDYJ8u49c10/nlLCtER9p/cmJqmc+fOHDhwgPj4eFq0aMHQoUO5+uqrSUlJoXv37pxxxhkn/Zx33XUXd9xxB127diUsLIw333yTyMhIJk2axLvvvkt4eDhxcXH87ne/Iz09nYcffhiPx0N4eDivvvpqlRyX3Q/iNPhkUQ4PTl5Cr1aNmDCiNzFR4W6XZEyNYfeDqLhqdT8IEblcRFaLSJaIPBpg/QgRyRWRxc7PbX7rbhGRNc7PLcGsM9iu6ZHAizf1ZNHGvdw8fj778orcLskYY04oaOc7RMQLvAz0B3KAdBH5VFVXltt0kqreXW7fxsDvgRRAgQXOvnuCVW+wXdmtBeFe4e73FzHkn3N5Z9TZNK4b4XZZxhgXLFu2jGHDhh21LDIyknnz5rlUUWDBPCGeCmSpajaAiEwEBgLlAyKQy4Bpqrrb2XcacDnwQZBqPS0u7RzH2OG9GPPOAm4aO5d3bzub2JhIt8syJuSpakjNg9a1a1cWL158Wl+zMt0JwTzFFA9s8nuc4ywr7zoRWSoiH4pI4snsKyKjRSRDRDJyc3Orqu6g6texGW+M6M3G3XkMGjuHbftO7tI3Y8zRoqKi2LVrl82Ddhyqyq5du4iKijqp/YLZgggU5+Xfwf8CH6hqgYjcAbwFXFTBfVHVscBY8HVSn1q5p8857Zry9qhURr6Rzo2vz+H9288moVG022UZE5ISEhLIyckhVL4kuiUqKoqEhIST2ieYAZEDJPo9TgC2+G+gqrv8Ho4D/ua3b79y+86o8gpd1Du5Me+MSuWWCfMZ9Ppc3r/9bFo1qet2WcaEnPDwcFq3bu12GTVSME8xpQPtRaS1iEQAg4FP/TcQkRZ+DwcAq5y/pwKXikgjEWkEXOosq1F6JDXi/dv7kFdYzI2vz2Ft7kG3SzLGmMOCFhCqWgzcje+DfRUwWVVXiMgTIjLA2exeEVkhIkuAe4ERzr67gSfxhUw68ERZh3VN0yW+ARNHp1FSqgx6fS6rtx1wuyRjjAFsoFy1kbXjIEP/OZfC4lLeGXU2XeIbuF2SMaYWcG2gnKm4ds3qMXlMGtERYQwZN5dFG0N2yIcxpoawgKhGWjWpy6QxfWgYHcGw8fNJX18jz6oZY0KEBUQ1k9Aomslj0mhWP5Lh4+fzfdZOt0syxtRSFhDVUFyDKCaNTiOpcTQj30xnxuodbpdkjKmFLCCqqdiYSD4Y3Yd2zeox+u0FfLlim9slGWNqGQuIaqxx3Qjev60PnVrW5673FjJl6Va3SzLG1CIWENVcg+hw3h2VSo+khtzzwUI+WZTjdknGmFrCAiIExESF89atqfRp04QHJi9h4vyNbpdkjKkFLCBCRHREGBNG9KZv+1ge/XgZb89Z73ZJxpgazgIihESFexk7vBf9z2zO7/6zgnEzs90uyRhTg1lAhJjIMC+vDO3JlV1b8KfPV/HS9DVul2SMqaGCOd23CZJwr4d/DO5OZJiHv3+ZSUFxKQ/07xBSd9QyxlR/FhAhKszr4ekbziIizMOL07MoKC7lsZ+dYSFhjKkyFhAhzOsR/nxNVyLCPIydmU1BUQm/v7ozHo+FhDHm1FlAhDiPR/jDgM5EhnkYN2sdBcWl/PmarhYSxphTZgFRA4gIv7qiE1HhXl6cnkVhcSlPXd+NMK9dg2CMqTwLiBpCRHjw0o5EeD08My2TgpJSnh/UnXALCWNMJVlA1DD3XNyeqHAvf/p8FYXFpbw0pAeRYV63yzLGhCD7elkD3d63DU8M7My0ldsZ884C8otK3C7JGBOCLCBqqOFpyfz12q58m5nLrW+mk1dY7HZJxpgQYwFRgw1OTeKZG85ibvYubpkwnwP5RW6XZIwJIRYQNdy1PRN44aYeLNq4l2Hj57PvRwsJY0zFBDUgRORyEVktIlki8uhxtrteRFREUpzH4SLylogsE5FVIvJYMOus6a7q1pJXhvZk5Zb9DBk3l92HCt0uyRgTAoIWECLiBV4GfgacCdwkImcG2C4GuBeY57f4BiBSVbsCvYAxIpIcrFprg0s7xzF2eC+ydhzkprFzyT1Q4HZJxphqLpgtiFQgS1WzVbUQmAgMDLDdk8BTQL7fMgXqikgYUAcoBPYHsdZaoV/HZrwxojcbd+cxaOwctu3LP/FOxphaK5gBEQ9s8nuc4yw7TER6AImq+lm5fT8EDgFbgY3A31V1d/kXEJHRIpIhIhm5ublVWnxNdU67prw9KpUd+wu48fU55OzJc7skY0w1FcyACDQZkB5eKeIBngMeDLBdKlACtARaAw+KSJufPJnqWFVNUdWU2NjYqqm6Fuid3Jh3RqWyN6+QQa/PZcOuQ26XZIyphoIZEDlAot/jBGCL3+MYoAswQ0TWA32AT52O6iHAF6papKo7gO+AlCDWWuv0SGrE+7f3Ia+wmBtfn8Pa3INul2SMqWaCGRDpQHsRaS0iEcBg4NOylaq6T1WbqmqyqiYDc4EBqpqB77TSReJTF194/BDEWmulLvENmDg6jZJSZdDrc1m97YDbJRljqpGgBYSqFgN3A1OBVcBkVV0hIk+IyIAT7P4yUA9Yji9o3lDVpcGqtTbrGBfDxNFpeD0weOwclm/e53ZJxphqQlT1xFuFgJSUFM3IyHC7jJC1Ydchhoybx4H8It66NZUeSY3cLskYcxqIyAJVDXgK30ZSGwBaNanLpDF9aBgdwbDx80lf/5OLxowxtYwFhDksoVE0k8ek0ax+JMPHz+f7rJ1ul2SMcZEFhDlKXIMoJo1OI6lxNCPfTGfG6h1ul2SMcYkFhPmJ2JhIPhjdh3bN6jH67QV8uWKb2yUZY1xgAWECalw3gvdv60OnlvW5672FTFm61e2SjDGnmQWEOaYG0eG8OyqVHkkNueeDhXyyKMftkowxp5EFhDmumKhw3ro1lT5tmvDA5CVMSt/odknGmNPEAsKcUHREGBNG9KZv+1ge+WgZb89Z73ZJxpjTwALCVEhUuJexw3vR/8zm/O4/K/jnrGy3SzLGBJkFhKmwyDAvrwztyZVdW/DHKat4afoat0syxgRRmNsFmNAS7vXwj8HdiQzz8PcvMykoLuWB/h0QCTS7uzEmlFlAmJMW5vXw9A1nERHm4cXpWRQUl/LYz86wkDCmhrGAMJXi9Qh/vqYrEWEexs7MpqCohN9f3RmPx0LCmJrCAsJUmscj/GFAZyLDPIybtY6C4lL+fE1XCwljaggLCHNKRIRfXdGJqHAvL07PorC4lKeu70aY165/MCbUWUCYUyYiPHhpRyK8Hp6ZlklBSSnPD+pOuIWEMSHNAsJUmXsubk9UuJc/fb6KwuJSXhrSg8gwr9tlGWMqyb7imSp1e982PDGwM9NWbmfMOwvILypxuyRjTCVZQJgqNzwtmb9e25VvM3O59c108gqL3S7JGFMJFhAmKAanJvHMDWcxN3sXt0yYz4H8IrdLMsacJAsIEzTX9kzghZt6sGjjXoaNn8++Hy0kjAklFhAmqK7q1pJXhvZkxZZ9DBk3lz2HCt0uyRhTQUENCBG5XERWi0iWiDx6nO2uFxEVkRS/Zd1EZI6IrBCRZSISFcxaTfBc2jmOscNTyNpxkMFj55J7oMDtkowxFRC0gBARL/Ay8DPgTOAmETkzwHYxwL3APL9lYcC7wB2q2hnoB9j5iRB2YcdmTBjRm4278xg0dg7b9uW7XZIx5gSC2YJIBbJUNVtVC4GJwMAA2z0JPAX4f2JcCixV1SUAqrpLVe16yRB3brumvHVrKjv2FzBo7Bxy9uS5XZIx5jiCGRDxwCa/xznOssNEpAeQqKqfldu3A6AiMlVEForI/wWxTnMapbZuzDujUtlzqJBBr89lw65DbpdkjDmGYAZEoBnb9PBKEQ/wHPBggO3CgPOAoc7va0Tk4p+8gMhoEckQkYzc3NyqqdoEXY+kRrx/ex/yCou58fU5rM096HZJxpgAghkQOUCi3+MEYIvf4xigCzBDRNYDfYBPnY7qHOBbVd2pqnnA50DP8i+gqmNVNUVVU2JjY4N0GCYYusQ3YOLoNEpKlUGvz2X1tgNul2SMKadCASEivxSR+uIz3jntc+kJdksH2otIaxGJAAYDn5atVNV9qtpUVZNVNRmYCwxQ1QxgKtBNRKKdDusLgJWVOD5TjXWMi2Hi6DS8Hhg8dg7LN+9zuyRjjJ+KtiBuVdX9+DqPY4GRwF+Pt4OqFgN34/uwXwVMVtUVIvKEiAw4wb57gGfxhcxiYKGqTqlgrSaEtGtWj8lj0oiOCGPIuLks2rjH7ZKMMQ5R1RNvJLJUVbuJyD+AGar6iYgsUtUewS+xYlJSUjQjI8PtMkwl5ezJY8i4eew+VMgbI3vTO7mx2yUZUyuIyAJVTQm0rqItiAUi8iVwBTDVGbtQWlUFGpPQKJrJY9JoVj+S4ePn833WTrdLMqbWq2hAjAIeBXo7ncbh+E4zGVNl4hpEMWl0GkmNoxn5ZjpfrdzudknG1GoVDYg0YLWq7hWRm4HfANajaKpcbEwkH4zuQ7tm9bjt7QxumTCfxZv2ul2WMbVSRQPiVSBPRM4C/g/YALwdtKpMrda4bgT/uiONRy4/gyU5e/n5y99x21vpdpWTMadZRQOiWH292QOBf6jqP/CNYzAmKKIjwrizX1tm/d+FPNi/A/PX7eaqF2dzxzsL+GHbfrfLM6ZWqOg9qQ+IyGPAMOB8ZyK+8OCVZYxPTFQ491zcnuHnJDN+9jomzF7H1JXbuLJrC+67pD3tmtn3FGOCpaKXucYBQ4B0VZ0lIklAP1WtNqeZ7DLX2mFvXiHjZmXzxnfryS8qYWD3eO69uD2tm9Z1uzRjQtLxLnOtUEA4T9Ic6O08nK+qO6qoviphAVG77DpYwNiZ2bw1Zz1FJcq1PXxBkdg42u3SjAkppxwQInIj8DQwA98kfOcDD6vqh1VY5ymxgKiddhzI59UZa3lv3kZKS5UbUhK5+6J2xDes43ZpxgRPaQlsXwEb58LGORDVAK5+vlJPVRUBsQToX9ZqEJFY4CtVPatSFQWBBUTttm1fPi9/k8XE9I0IwuDURO7q1464BnYjQlMDFP0Imxf4wmDjXNg0HwqcizViWsIZV8CVz1TqqasiIJapale/xx5gif8yt1lAGIDNe3/kpelZ/CtjEx6PMPTsJO7s15ZmMRYUJoQc2gWb5sHG732BsGUxlDo31Wx2JiT1gaQ03+8GiSCB7q5QMVUREE8D3YAPnEWD8N3x7ZFKV1XFLCCMv4278nhx+ho+XrSZcK9wS1oyo/u2oUm9SLdLM+ZoqrB3A2yYc6SFsHO1b503Alr2PBIIiakQXbXzlFVVJ/V1wLn4+iBmquonVVfiqbOAMIGs23mIF75ew78Xb6ZOuJeR5yZz+/ltaBgd4XZpprYqLYHty4/0H2ycCwe2+tZFNoCks48EQsueEB7c1m+VBER1ZwFhjidrxwGe+2oNU5ZuJSYyjFvPa82t57WmQR0bzmOCrDDP6T9wAmHTfCh0bpBVP/7IqaJW50BsJ/AE8z5uP1XpgBCRA/jdJtR/FaCqWr9qSjx1lQ4IVciY4Duv17wzRFWbQzJB8MO2/Tw/bQ1frNhG/agwbj+/DSPPa029yIqOGTXmBA7tgk1+rYOA/Qfn+H43TDz+c50G1oI4nr0b4Xm/vvaGrSCuKzTvAnFdfL8bJZ9SJ5CpfpZv3sfzX2Xy1aodNIoOZ3TfttxyTiuiIywozElQhT3rndaB06G8M9O3zhsB8b2O7j+o08jVcgOxgDgeVdi/xXdOcNsy38/25bBrLYcbTxExvtZFXBcnPLpCs04QYYOyQt3iTXt5blom32bm0rReBHdc0Jab+7QiKtzrdmmmOirrP/DvUD64zbcuqgEk9vHrP+gR9P6DqmABURmFh2DHqiOBsW25b2BK2blD8UDjtkdaGWWtjvotrbURgjLW7+a5rzL5LmsXzWIiuatfWwanJllQ1HaFebA5w6//IP3IZ0CDRCcMnFNGsWec9v6DqmABUVVKS32Xox1ubSyH7ct8p6nK1Gl0JDDKQiO2I4TZ5ZWhYG72Lp79MpP563fTokEUv7iwHTemJBIRFnr/45tKOLTz6KuLti6G0mJAjvQftDoHEs+uFv0HVcECItjy9/laF2WBsW057FgJxfm+9Z4waNrRr7XRxXeaql6sO/Wa41JVvsvaxTPTVrNo417iG9bh3ovbcW3PBMK9FhQ1hirsWecLgg1O/8GuNb513shy/Qe9q2X/QVWwgHBDaYmvH6MsMMpaHWXXOwPUa/7T1kaTduC1jtLqQFWZkZnLc9MyWZqzj1ZNorn3ovb8vEc8Xo+dRgw5JcXO+AP//gPntrZRDY5cbpqUBi26h0T/QVWwgKhODu0qFxrLIfeHI5fBhUX5zmWWtTLKWh11Grpbdy2mqny1agfPTstk1db9tImty32XdOCqri3wWFBUX4WHIMev/yAnHQoP+tY1SPLrP0gL2f6DqmABUd0VF/oujStrZZT9ztt1ZJsGiUdOT5W1Nhq1rrX/qN1QWqp8uXIbz01bw+rtB+jQvB73XdKByzvHWVBUBwdznfEHTiBsXXKk/6B553LzFyW4XW214VpAiMjlwD8AL/BPVf3rMba7HvgX0FtVM/yWJwErgcdV9e/He62QDohAVOHAtnKhsdx3jlRLfdtE1PN1nPlfSdXsTIis527tNVxpqTJl2Vae/yqTtbmH6NSiPvdf0p7+ZzZH7Aq200MVdmf7dSjPgV1ZvnX+/QetzoGE3tYCPw5XAsK5LWkm0B/IAdKBm1R1ZbntYoApQARwd7mA+AgoBebVuoA4lqIffZffHr701vldsM/ZQKBx69oOy4UAABaMSURBVJ/2bTRIsMtvq1hJqfKfxZv5x9dr2LArj67xDXigfwf6dYy1oKhqJcW+U7P+4w8OOfcsi2p4dP9By+521eBJOF5ABLM3NBXIUtVsp4iJwEB8LQJ/TwJPAQ/5LxSRnwPZwKEg1hh6wutAfE/fTxlV36W2h0PDGfC36tMj20Q1OLpPI66Lb96XWtIRFwxej3BtzwQGnNWSjxdt5oWv1zDyzXR6JDXkgf4dOK9dUwuKyio85Osz8B9/UOR8FDRMgrYXHgmEph3tVGuQBDMg4oFNfo9zgLP9NxCRHkCiqn4mIg/5La8LPIKv9XFUcJTbfzQwGiApKanqKg81ItCole/njCuPLC84ANtXHt0pvvBtKMpz9vNC0/Z+rQ2nYzymuTvHEaLCvB5uTEnk593j+XBBDi9NX8Ow8fNJTW7M/f07kNa2idslVn8Hdzhh4Nd/oCX4+g+6QPchRwKhQbzb1dYawQyIQF+dDp/Pcm469BwwIsB2fwCeU9WDx/sGpqpjgbHgO8V0KsXWSJExztTBfrlcWuq79nvb0iOhsXEuLPe7e2zd2KPHa8R1gaYdwGsznx5PRJiHIWcncV2veCalb+Kl6VncNG4u57RtwgP9O5CSXLXz+Iesw/0HzumiDXNg91rfOm8kJKTAeff5Ricn9va1fo0rgtkHkYavc/ky5/FjAKr6F+dxA2At4Fx3RhywGxiALzjKhik2xNcP8TtVfelYr1dr+iCCJW+3b7Cf/2mqHT9ASYFvvTfCNyK8eVe/1kaXKr95SU2SX1TCe/M28uqMLHYeLKRvh1ge6N+B7om1rMO0pNj3hcR/QrtDub51dRr55i9qleaMPzjL+g9OM7c6qcPwdVJfDGzG10k9RFVXHGP7GcBD/p3UzvLHgYPWSe2CkiLYueanV1KVdQ6Cbz57/5lv47pC4zbgsTmMyuQVFvPOnA289u1a9uQVcfEZzbi/fwe6xNfQb8YFB4/uP8jJ8Os/aHV0h3LTDtZ/4DJXOqlVtVhE7gam4rvMdYKqrhCRJ4AMVf30+M9gXOcNh+Zn+n663Xhk+cEd5SYxXA5ZXznnjIHwaN9st0ddSdXZd8qrFoqOCGPMBW0Z2qcVb32/nrEzs7nqxdlc1rk5913SgU4tQvweJAd3HLmyaOMc2Lr0SP9BXBfoMdQXCIl9rP8gxNhAOVM1igt8I8K3lZtaJH/vkW0aJR89821cF983ylp2pc/+/CImzF7H+FnrOFBQzJVdW3DfJe1p3zwEAlTVN4WMfyCU9R+ERUF8ytHzF1n/QbVnI6mNO1Rh/+ajJzEsf6+NyPq+1oX/KPFmZ/ou5z0d9Wmpb94sLfGNui0tKbfM/3fVLs8rKGDW6u3MydpBSUkJ3eNjuKBDY5pGh/ltXxpg/2MsD/hax3qOk6nZb/+iH4+MuanTyO900TlO/4Hd6zvUWECY6uUn99pY5txrw7leQTy+SQsbJPh9YJ3gg6syH4hlI9JDhXh9fTtH/fYEWO45/naesIpvW365N8IX4klp0KS99R/UAG4NlDMmsIi6vksZE/z+TZaWwt71R48OP7jt6A+osIgKfhhWdHlYFTxHBT5UK7h856Fi3pizkQ8yNlNUKvy8ZyJj+nUgoXGMfRAbV1gLwphqZvv+fF75JosP5m9CUQb1TuTuC9sT18BGvZuqZ6eYjAlBm/f+yMvfZDE5fRMejzAkNYm7LmxLsxgLClN1LCCMCWGbdufx4vQ1fLRwM+FeYXhaMmP6tqFJPRtQZk6dBYQxNcD6nYd44es1/HvxZqLCvYw4J5nbz29Do7p25ZCpPAsIY2qQrB0H+cfXa/hs6RbqRoRx67nJjDq/DQ3q2FxZ5uRZQBhTA63edoDnv8rkf8u3ERMVxu3nt2HkucnERFlQmIqzgDCmBluxZR/PTVvDV6u20zA6nNF923BLWjJ1I+0qdnNiFhDG1AJLc/by7LRMZqzOpUndCO64oC0392lFnQibONEcmwWEMbXIgg17eP6rTGat2UlsTCR39WvLTalJRIVbUJifsoAwphaal72LZ6dlMm/dbuLqR/GLi9oxKCWRiDAblW2OsIAwppZSVeas3cUz0zJZsGEP8Q3rcM9F7biuVwLhXgsKYwFhTK2nqsxcs5Nnv1zNkpx9JDWO5t6L2/Pz7i0Js6Co1SwgjDGALyim/7CDZ6dlsmLLfto0rcsvL2nPVd1a4vXUrvtyGJ/jBYR9dTCmFhERLu7UnM/uOY/Xbu5FRJiHX05czOXPz2TK0q2UltaML4ymalhAGFMLiQiXd4nj83vP56UhPVDgF+8v5IoXZjF1xTZqypkFc2osIIypxTwe4apuLZl6X1+eH9SdguJSxryzgKtfms30H7ZbUNRy1gdhjDmsuKSUTxZt5oXpa9i0+0e6Jzbkgf4dOL99U6SW3Tu8trBOamPMSSkqKeXDBTm8+PUatuzLp3dyI+7v34Fz2jZ1uzRTxSwgjDGVUlBcwuT0Tbz0TRbb9xeQ1qYJ91zUjrS2TaxFUUO4dhWTiFwuIqtFJEtEHj3OdteLiIpIivO4v4gsEJFlzu+LglmnMSawyDAvw9KS+fbhC/n91WeSlXuQIf+cx89f+Z4vlm+zq55quKC1IETEC2QC/YEcIB24SVVXltsuBpgCRAB3q2qGiPQAtqvqFhHpAkxV1fjjvZ61IIwJvvyiEj5amMPr32azcXcebWPrcscFbRnYPd6m8AhRbrUgUoEsVc1W1UJgIjAwwHZPAk8B+WULVHWRqm5xHq4AokTE7q9ojMuiwr0MPbsV0x+8gBdu6kFEmJeHP1xKv6e/YcLsdeQVFrtdoqlCwQyIeGCT3+McZ9lhTkshUVU/O87zXAcsUtWCqi/RGFMZYV4PA85qyef3nscbI3uT0CiaJz5bybl/nc4LX69hb16h2yWaKhDMO4oE6sE6fD5LRDzAc8CIYz6BSGfgb8Clx1g/GhgNkJSUdAqlGmMqQ0S4sGMzLuzYjIz1u3l1xlqenZbJ69+uZcjZSYw6rw1xDaLcLtNUUjD7INKAx1X1MufxYwCq+hfncQNgLXDQ2SUO2A0McPohEoDpwEhV/e5Er2d9EMZUDz9s289rM9by36Vb8Ypwbc94RvdtQ5vYem6XZgJw5TJXEQnD10l9MbAZXyf1EFVdcYztZwAPOeHQEPgWeEJVP6rI61lAGFO9bNqdx9iZ2UzK2ERRSSlXdGnBnf3a0iW+gdulGT+udFKrajFwNzAVWAVMVtUVIvKEiAw4we53A+2A34rIYuenWbBqNcZUvcTG0Tz58y5898hF3HlBW2Zm5nLVi7MZNn4ec9busmk8QoANlDPGnBb784t4d+4GJsxex86DhXRPbMhd/dpySafmeGyqcdfYSGpjTLWRX1TCvxbkMHbmWjbt/pH2zepxxwVtGdC9pd3lzgUWEMaYaqe4pJQpy7by6oy1/LDtAPEN63D7+a0Z1DuJOhFet8urNSwgjDHVlqryzeodvPLNWjI27KFx3QhGnpPM8LRkGkSHu11ejWcBYYwJCenOWIrpP+ygboSXoX1aMeq81jSvb2MpgsUCwhgTUlZt3c+rM9by2dIthHk8XNcrnjF925LctK7bpdU4FhDGmJC0cVcer89cy78W5FBcUsrPurbgzgtsLEVVsoAwxoS0HQfymTB7Pe/O3cDBgmIu6BDLnf3acnbrxnZfilNkAWGMqRH2/egbS/HGd76xFD2TGnJnv3ZcfEYzG0tRSRYQxpgaJb+ohH9lbOL1mdnk7PmRDs19YymuPsvGUpwsCwhjTI1UXFLKZ0t9YylWb/eNpRjdtw03piTaWIoKsoAwxtRopaXOWIoZa1mwYQ9N6kYw8txkhqUl06COjaU4HgsIY0ytoKqkr9/DKzOymLE6l3qRYQw9O4lR57WmmY2lCMgCwhhT66zYso/Xvs1mytIthHk9XN8rgTF929CqiY2l8GcBYYyptdbvPMTYWdl8mJFDcWkpV3ZryR0XtKFzSxtLARYQxhjDjv35jP9uHe/N3cjBgmL6dYzlrn7t6J3cqFaPpbCAMMYYR9lYigmz17HrUCG9WjXirn5tubBj7RxLYQFhjDHl/FhYwuSMTYydmc3mvT/SsXkMd/Zry1XdWhBWi8ZSWEAYY8wxFJWU8tnSLbw6Yy2Z2w+S0KgOY/q24YaURKLCa/5YCgsIY4w5gdJS5esfdvDKjCwWbdxL03oRjDy3NcPSWlE/quaOpbCAMMaYClJV5q3z3Zfi28xcYiLDGNqnFbeel0yzmJo3lsICwhhjKmH55n28+u1a/rdsK2FeDzf0SmBM37YkNYl2u7QqYwFhjDGnYN3OQ4yduZaPFmymuLSUq7q15M5+benUor7bpZ0yCwhjjKkC2/fnM372Ot6bu4FDhSVcdEYz7uzXlt7Jjd0urdKOFxBBvZZLRC4XkdUikiUijx5nu+tFREUkxW/ZY85+q0XksmDWaYwxFdG8fhS/uqIT3z96MQ/278DiTXu54bU5XP/q90z/YTs15Qt3maC1IETEC2QC/YEcIB24SVVXltsuBpgCRAB3q2qGiJwJfACkAi2Br4AOqlpyrNezFoQx5nT7sbCESekbGTdrHZv3/sgZcb6xFFd2DZ2xFG61IFKBLFXNVtVCYCIwMMB2TwJPAfl+ywYCE1W1QFXXAVnO8xljTLVRJ8LLiHNbM+Phfjxzw1mUlCq/nLiYC5+ZwTtzN5BfdMzvtCEhmAERD2zye5zjLDtMRHoAiar62cnu6+w/WkQyRCQjNze3aqo2xpiTFO71cF2vBKbe15exw3rRpG4kv/33cs772ze8MiOL/flFbpdYKcEMiECTmhw+nyUiHuA54MGT3ffwAtWxqpqiqimxsbGVLtQYY6qCxyNc2jmOT+46hw9u70OnFjE89cVqzv3LdJ764gdyDxS4XeJJCQvic+cAiX6PE4Atfo9jgC7ADGcmxTjgUxEZUIF9jTGm2hIR0to2Ia1tE99YihlrefXbtYyfvY4bUxIZ3bcNiY2r/1iKYHZSh+HrpL4Y2Iyvk3qIqq44xvYzgIecTurOwPsc6aT+GmhvndTGmFCVnXuQsTOz+WhhDqUKV3drwR392nJGnLtjKY7XSR20FoSqFovI3cBUwAtMUNUVIvIEkKGqnx5n3xUiMhlYCRQDvzheOBhjTHXXJrYef72uG/dd0oHxs7N5b95G/r14Cxc7YylSquFYChsoZ4wxLtibV8jbczbwxnfr2JNXRGpyY+68sC39OsSe1hsY2UhqY4yppvIKi5mUvolxM7PZsi+fTi3qc2e/tlzRJe60jKWwgDDGmGqusLiU/yzezGvfrmVt7iGSGkcz5oI2XNczIaj3pbCAMMaYEFFaqny5cjuvzshiSc4+YmMiGXVea4aenURMEO5LYQFhjDEhRlWZs3YXr367lllrdhITFcbwtFaMPLc1TetFVtnrWEAYY0wIW5qzl9e+Xcv/lm8jwuthUO9Ebj+/asZSWEAYY0wNsDb3IGO/zebjRb6xFAPOaskdF7SlY1xMpZ/TAsIYY2qQrft+ZPysdbw/fyN5hSVc2a0FL93Uo1KXx7oyUM4YY0xwtGhQh99cdSa/uLAdb8/ZQGFJSVDGTlhAGGNMiGpUN4JfXtI+aM8fGne0MMYYc9pZQBhjjAnIAsIYY0xAFhDGGGMCsoAwxhgTkAWEMcaYgCwgjDHGBGQBYYwxJqAaM9WGiOQCG07hKZoCO6uoHDfVlOMAO5bqqqYcS005Dji1Y2mlqrGBVtSYgDhVIpJxrPlIQklNOQ6wY6muasqx1JTjgOAdi51iMsYYE5AFhDHGmIAsII4Y63YBVaSmHAfYsVRXNeVYaspxQJCOxfogjDHGBGQtCGOMMQFZQBhjjAmoVgWEiFwuIqtFJEtEHg2wPlJEJjnr54lI8umvsmIqcCwjRCRXRBY7P7e5UeeJiMgEEdkhIsuPsV5E5AXnOJeKSM/TXWNFVeBY+onIPr/35Henu8aKEJFEEflGRFaJyAoR+WWAbULifangsYTK+xIlIvNFZIlzLH8IsE3Vfoapaq34AbzAWqANEAEsAc4st81dwGvO34OBSW7XfQrHMgJ4ye1aK3AsfYGewPJjrL8C+B8gQB9gnts1n8Kx9AM+c7vOChxHC6Cn83cMkBng31dIvC8VPJZQeV8EqOf8HQ7MA/qU26ZKP8NqUwsiFchS1WxVLQQmAgPLbTMQeMv5+0PgYgnGjV5PXUWOJSSo6kxg93E2GQi8rT5zgYYi0uL0VHdyKnAsIUFVt6rqQufvA8AqIL7cZiHxvlTwWEKC89/6oPMw3Pkpf5VRlX6G1aaAiAc2+T3O4af/UA5vo6rFwD6gyWmp7uRU5FgArnOa/x+KSOLpKa3KVfRYQ0Wac4rgfyLS2e1iTsQ5RdED37dVfyH3vhznWCBE3hcR8YrIYmAHME1Vj/m+VMVnWG0KiEApWj59K7JNdVCROv8LJKtqN+ArjnyrCDWh8p5UxEJ8896cBbwI/Nvleo5LROoBHwH3qer+8qsD7FJt35cTHEvIvC+qWqKq3YEEIFVEupTbpErfl9oUEDmA/7foBGDLsbYRkTCgAdXzlMEJj0VVd6lqgfNwHNDrNNVW1SryvoUEVd1fdopAVT8HwkWkqctlBSQi4fg+UN9T1Y8DbBIy78uJjiWU3pcyqroXmAFcXm5VlX6G1aaASAfai0hrEYnA14HzabltPgVucf6+HpiuTm9PNXPCYyl3PngAvnOvoehTYLhz1UwfYJ+qbnW7qMoQkbiy88Eikorv/79d7lb1U06N44FVqvrsMTYLifelIscSQu9LrIg0dP6uA1wC/FBusyr9DAur7I6hRlWLReRuYCq+q4AmqOoKEXkCyFDVT/H9Q3pHRLLwpe5g9yo+tgoey70iMgAoxncsI1wr+DhE5AN8V5E0FZEc4Pf4Ot9Q1deAz/FdMZMF5AEj3an0xCpwLNcDd4pIMfAjMLiafgE5FxgGLHPOdwP8CkiCkHtfKnIsofK+tADeEhEvvhCbrKqfBfMzzKbaMMYYE1BtOsVkjDHmJFhAGGOMCcgCwhhjTEAWEMYYYwKygDDGGBOQBYQxJyAiJX4zfS6WALPnnsJzJx9r9ldj3FZrxkEYcwp+dKY3MKZWsRaEMZUkIutF5G/OHP3zRaSds7yViHztTJT4tYgkOcubi8gnzqRwS0TkHOepvCIyzpnj/0tnlCwicq+IrHSeZ6JLh2lqMQsIY06sTrlTTIP81u1X1VTgJeB5Z9lL+KbC7ga8B7zgLH8B+NaZFK4nsMJZ3h54WVU7A3uB65zljwI9nOe5I1gHZ8yx2EhqY05ARA6qar0Ay9cDF6lqtjMh3DZVbSIiO4EWqlrkLN+qqk1FJBdI8JtEsWwK6mmq2t55/AgQrqp/FJEvgIP4Zhf9t9+9AIw5LawFYcyp0WP8faxtAinw+7uEI32DVwIv45uJd4EzO6cxp40FhDGnZpDf7znO399zZJK0ocBs5++vgTvh8I1f6h/rSUXEAySq6jfA/wENgZ+0YowJJvtGYsyJ1fGbCRTgC1Utu9Q1UkTm4fuydZOz7F5ggog8DORyZKbTXwJjRWQUvpbCncCxpsj2Au+KSAN8N4F5zrkHgDGnjfVBGFNJTh9EiqrudLsWY4LBTjEZY4wJyFoQxhhjArIWhDHGmIAsIIwxxgRkAWGMMSYgCwhjjDEBWUAYY4wJ6P8BUqeeyG76R/gAAAAASUVORK5CYII=)







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
  vocab_size = len(word_idx) + 1
  embedding_dim = 200

  embedding_matrix = np.zeros((vocab_size, embedding_dim))
  ko_model= Word2Vec.load('new_word2vec_movie.model')

  for word, idx in tokenizer.word_index.items():
      embedding_vector = ko_model[word] if word in ko_model else None
      if embedding_vector is not None:
          embedding_matrix[idx] = embedding_vector

  return training_padded, testing_padded, training_labels,testing_labels,embedding_matrix, vocab_size
```

### word2vec weight 만들기(위의 함수에 포함된 부분)

tokenizer에 있는 단어 사전을 순회하면서 word2vec의 200차원 vector를 가져오기

```python
  # word2vec weight 
  vocab_size = len(word_idx) + 1 # OOV를 포함하고 있기 때문에 1을 더해준다.
  embedding_dim = 200

  embedding_matrix = np.zeros((vocab_size, embedding_dim))
  ko_model= Word2Vec.load('word2vec_movie.model')

  for word, idx in tokenizer.word_index.items():
      embedding_vector = ko_model[word] if word in ko_model else None
      if embedding_vector is not None:
          embedding_matrix[idx] = embedding_vector
            
```

```python
vocab_size
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

![model2_accuracy.png](https://github.com/hamakim94/CNN_Project/blob/master/result_file/model2_accuracy.png?raw=true)

![model2_loss.png](https://github.com/hamakim94/CNN_Project/blob/master/result_file/model2_loss.png?raw=true)







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
max_length = 53         (문장 최대 길이는 53으로)
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
  history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels))

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

![model3_accuracy.png](https://github.com/hamakim94/CNN_Project/blob/master/result_file/model3_accuracy.png?raw=true)

epoch가 커질수록 train_data에 대한 accuracy가 높아지지만, overfit 된다

![model3_loss.png](https://github.com/hamakim94/CNN_Project/blob/master/result_file/model3_loss.png?raw=true)

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

실제값:0, 예측값:[0.30392236]
실제값:0, 예측값:[0.0650005]
실제값:1, 예측값:[0.21971333]
실제값:1, 예측값:[0.99392986]
실제값:1, 예측값:[0.6215658]
실제값:0, 예측값:[0.83809507]
실제값:0, 예측값:[0.22184402]
실제값:1, 예측값:[0.68833643]
실제값:1, 예측값:[0.8921679]
실제값:0, 예측값:[0.39271945]
실제값:1, 예측값:[0.78049505]
실제값:1, 예측값:[0.4866558]
실제값:0, 예측값:[0.08552262]
실제값:1, 예측값:[0.51187897]
실제값:1, 예측값:[0.45225626]
실제값:1, 예측값:[0.03050807]
1/1 [==============================] - 0s 0s/step - loss: 0.7262 - accuracy: 0.6875
정답률은 68.75% 입니다.



test 값은 model1 : 40%대, model2 : 50%대로 model3가 제일 높았다

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