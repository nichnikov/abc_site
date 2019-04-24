#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:40:59 2018

@author: alexey
"""

# LSTM for sequence classification in the IMDB dataset
import os, pickle, TextsHandling, keras
#import pandas as pd
import numpy as np
from gensim import models, logging

def fragment2tensor (fragment_list, w2v, tensor_length = 150, vec_len = 200):
    try:
        vectors_list = w2v[fragment_list[:tensor_length]]
        if vectors_list.shape[0] < tensor_length:        
            add_v = np.array((tensor_length-vectors_list.shape[0])*[[0]*vec_len])
            tensor = np.concatenate((np.array(vectors_list), add_v), axis = 0)
        else:
            tensor = np.array([vectors_list])
    except:
        tensor = np.array(tensor_length*[[0]*vec_len])
    return tensor    

def flatten_all (iterable):
	for elem in iterable:
		if not isinstance(elem, list):
			yield elem
		else:
			for x in flatten_all(elem):
				yield x

#model_rout = r'/home/alexey/Dropbox/abc_site/models'
#data_rout = r'/home/alexey/Dropbox/abc_site/data/'
#dict_rout = r'/home/alexey/Dropbox/abc_site/dicts'

model_rout = r'/home/an/Dropbox/abc_site/models'
dict_rout = r'/home/an/Dropbox/abc_site/dicts'

#лемматизация текстов:
text  = 'Раскрутка сайта www.c-vertical.ru.Занимаюсь подбором ключевых словосочетаний для раскрутки сайта. Подскажите из скольки слов должно быть словосочетание. Например: &quot;строительство зданий из'
#text = ''

#лемматизация полученного текста:
lemm_texts = TextsHandling.text_coll_lemmatizer([text])

#загрузка модели word2vec
w2v_model = models.Word2Vec.load(os.path.join(model_rout, 'w2v_model_abc_tech_20181130'))

#загрузка рабочего словоря (словоря модели word2vec)
with open(os.path.join(dict_rout, 'w2v_vocab.pickle'), 'rb') as f:
    work_vocab = pickle.load(f)

#оставим только слова из словаря модели (из рабочего словаря)
lemm_texts_true_words = [[w for w in sen.split() if w in work_vocab] for sen in lemm_texts]
w2v = models.Word2Vec.load(os.path.join(model_rout, 'w2v_model_abc_tech_20181130'))

#сформируем тензоры для обучения сверточной нейронной сети:
it = flatten_all(lemm_texts_true_words)

lemm_texts_for_tens = []
for w in it:
    lemm_texts_for_tens.append(w)

tensor = fragment2tensor(lemm_texts_for_tens, w2v, tensor_length = 150, vec_len = 150)
tensor3D=np.array([tensor]).reshape(1, 150, 150)

#загрузка классификатора (модели сверточной нейронной сети)
cnn_classifier = keras.models.load_model(os.path.join(model_rout, 'cnn_model_abc_site_20181130_tensor150.h5'))
result = cnn_classifier.predict_proba(tensor3D)
print(result)