from tensorflow import keras
import os
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot


doc = [
    '영화 졸라 재미없다',
    '돈아깝다 개같은거',
    '별점 0점짜리 쓰레기'
    '시간이 아깝다'
    '아주 재미지네요 하하하ㅏ'
]


checkpoint_dir = './ckpt1'

def restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]

    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print('Restoring from', latest_checkpoint)
    return keras.models.load_model(latest_checkpoint)


model = restore_model()
print(model.summary())
model.predi