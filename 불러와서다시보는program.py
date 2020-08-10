from tensorflow import keras
import os
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from functions import input_preprocessing, restore_model
import numpy as np

doc = [
    '영화 졸라 재미없다',
    '돈아깝다 개같은거',
    '별점 0점짜리 쓰레기',
    '시간이 아깝다',
    '아주 재미지네요 하하하ㅏ'
]
y= [0,0,0,0,1]

X = input_preprocessing(doc)
print(X)
model = restore_model('model1') #model1|model2|model3 입력하면 각각의 저장된모델중 가장 최신의것을 골라서 불러온다
y = np.array(y)
print(model.summary())

yhat = model.predict(X)
        # Compute the loss value for this batch.
for i in range(len(yhat)):
    print('실제값:{}, 예측값:{}'.format(y[i], yhat[i]))
result = model.evaluate(X,y)
print('정답률은 {}% 입니다.'.format(100*result[1]))