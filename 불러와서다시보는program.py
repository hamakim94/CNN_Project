from tensorflow import keras
import os
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from functions import input_preprocessing, restore_model


doc = [
    '영화 졸라 재미없다',
    '돈아깝다 개같은거',
    '별점 0점짜리 쓰레기'
    '시간이 아깝다'
    '아주 재미지네요 하하하ㅏ'
]
y= [0,0,0,0,1]

X = input_preprocessing(doc)

model = restore_model('model1')

print(model.summary())
model.evaluate(X,y)