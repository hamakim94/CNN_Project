import pickle
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import functions as fc

max_len = 30

df_train = pd.read_pickle('token_train_data.pkl')
df_test = pd.read_pickle('token_test_data.pkl')

token_train_data, train_lable = df_train['tokens'], df_train['labels']
token_test_data, test_lable = df_test['tokens'], df_test['labels']


final_train_data = fc.simple_fasttext_vectorize(fc.pad_sequence(token_train_data))
final_test_data = fc.simple_fasttext_vectorize(fc.pad_sequence(token_test_data))

print(len(final_test_data))
######
embedding_dim = 200
filter_sizes = (3, 4, 5)
num_filters = 100
dropout = 0.5
hidden_dims = 10
batch_size = 50
num_epochs = 10
min_word_count = 1
context = 10
conv_blocks = []
sequence_length = 200

model = fc.CNN_model_1()

import os #폴더 생성
checkpoint_dir = './ckpt1'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=0),
    # This callback saves a SavedModel every 100 batches.
    # We include the training loss in the folder name.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + '/ckpt-loss={loss:.3f}',
        save_freq=500)
]
history = model.fit(final_train_data, train_lable, epochs=10, callbacks=callbacks, batch_size = batch_size, validation_data=(final_test_data, test_lable))

#CNN 모델에 fit한 history, string: ('accuracy'or'loss'), name : 해당이름으로 차트title과 차트파일명이 됨
#result_file폴더에 결과 그래프 저장
fc.plot_graphs(history, 'accuracy',name='model1_acc')
fc.plot_graphs(history, 'loss', name='model1_loss')