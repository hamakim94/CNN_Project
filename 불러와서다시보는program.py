from tensorflow import keras
import os
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from functions import input_preprocessing, restore_model
import numpy as np

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