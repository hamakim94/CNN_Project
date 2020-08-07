import numpy as np
import pandas as pd
from konlpy.tag import Okt
from gensim import models

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

def fasttext_vectorize(padded_sentences):
    paddedarray = np.array([ ko_model.wv.word_vec(token) for x in padded_sentences for token in x])
    final_array=paddedarray.reshape(-1,max_len,300)
    return final_array