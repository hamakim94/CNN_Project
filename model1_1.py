import pickle
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import functions as fc
from functions import input_preprocessing, preprocessing, CNN_model_1, input_preprocessing2, ready_callbacks


if __name__ == "__main__":
    
    # #data 준비시키기 일반 version
    # df_train = pd.read_csv('train_data.csv')
    # df_test = pd.read_csv('test_data.csv')

    # train_data, train_lable = preprocessing(df_train)
    # test_data, test_lable = preprocessing(df_test)


    # X_train, y_train = input_preprocessing(train_data), train_lable
    # X_test, y_test = input_preprocessing(test_data), test_lable

    #data 준비시키기 시간절약 version
    df_train = pd.read_pickle('token_train_data.pkl')
    df_test = pd.read_pickle('token_test_data.pkl')

    train_data, train_lable = df_train['tokens'], df_train['labels']
    test_data, test_lable = df_test['tokens'], df_test['labels']

    X_train, y_train = input_preprocessing2(train_data), train_lable
    X_test, y_test = input_preprocessing2(test_data), test_lable

    #model 불러오기
    model = CNN_model_1(filter_sizes = (3, 4, 5),
                        num_filters = 100,
                        dropout = 0.5,
                        hidden_dims = 10,
                        max_len = 30)
        
    callbacks = ready_callbacks('ckpt1')

    # 모델 fit 시키기
    batch_size = 50
    num_epochs = 10
    history = model.fit(X_train, y_train, epochs=num_epochs, callbacks=callbacks, batch_size = batch_size, validation_data=(X_test, y_test))

    #CNN 모델에 fit한 history, string: ('accuracy'or'loss'), name : 해당이름으로 차트title과 차트파일명이 됨
    #result_file폴더에 결과 그래프 저장

    fc.plot_graphs(history, 'accuracy',name='model1_acc')
    fc.plot_graphs(history, 'loss', name='model1_loss')