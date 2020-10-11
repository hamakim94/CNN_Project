from functions import ready_embedding_matrix, input_preprocessing2, restore_model, LSTM_model, ready_callbacks, plot_graphs
import pandas as pd
import os

df_train = pd.read_pickle('token_train_data.pkl')
df_test = pd.read_pickle('token_test_data.pkl')

train_data, train_lable = df_train['tokens'], df_train['labels']
test_data, test_lable = df_test['tokens'], df_test['labels']

X_train, y_train = input_preprocessing2(train_data), train_lable
X_test, y_test = input_preprocessing2(test_data), test_lable

checkpoint_dir = 'LSTM'
#model 불러오기
if os.path.exists(checkpoint_dir):
    model = restore_model(checkpoint_dir)
    print('저장된 모델을 불러옵니다')
else : 
    embedding_matrix = ready_embedding_matrix()
    model = LSTM_model(embedding_matrix=embedding_matrix)
    print('---새로 학습을 시작합니다-----')


print(model.summary())
callbacks = ready_callbacks(checkpoint_dir)

# 모델 fit 시키기
batch_size = 50
num_epochs = 10
history = model.fit(X_train, y_train, epochs=num_epochs, callbacks=callbacks, batch_size = batch_size, validation_data=(X_test, y_test))

#CNN 모델에 fit한 history, string: ('accuracy'or'loss'), name : 해당이름으로 차트title과 차트파일명이 됨
#result_file폴더에 결과 그래프 저장

plot_graphs(history, 'accuracy',name='model1_acc')
plot_graphs(history, 'loss', name='model1_loss')