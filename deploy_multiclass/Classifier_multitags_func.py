#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:40:59 2018

@author: alexey
"""

# LSTM for sequence classification in the IMDB dataset
import os, pickle, TextsHandling, keras, glob, re
import pandas as pd
import numpy as np
from gensim import models, logging
import tensorflow as tf

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

def text2tensor(text):
    model_rout = r'./models'
    dict_rout = r'./dicts'

    #лемматизация полученного текста:
    lemm_texts = TextsHandling.text_coll_lemmatizer([text])

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
    
    return tensor3D


def classifier_cnn(text, cnn_classifier_list, tags_list, graph_defaut):
   #загрузка классификатора (модели сверточной нейронной сети)
   classes_list = []
   tensor3D = text2tensor(text)
   for cnn_classifier in cnn_classifier_list:       
       with graph_defaut.as_default():
           class_est = cnn_classifier.predict_proba(tensor3D)
           classes_list.append(str(class_est[0][1]))
   class_tags = list(zip(tags_list, classes_list))
   res = ' '.join([str(tg) + ':' + str(est) for tg, est in class_tags])
   return res#' '.join(classes_list)

#test:
if __name__ == '__main__':
    model_rout = r'./models'
    dict_rout = r'./dicts'


    text  = 'Редизайн сайта ориг.рф.Макет https://moqups.com/abvsait/iG6S5n1N/p:aff6727f3 Бриф https://docs.google.com/document/d/1uxRwbyijToOPG4B-zyRq66y0ISdXAP_vinZMH01Jw-A/pub Смета: Дизайн и верстка на основании вашей заготовки из архива 17 000 р Сборка на админке 8 000 р. с переносом контента  Итого 25 000 р'
    global graph_defaut
    graph_defaut = tf.get_default_graph()
    
    global cnn_classifier_list
    cnn_classifier_list = []
    
    global tags_list
    tags_list = []
    #fnms = ['cnn_model_abc_site_20181208_tensor150_tag105.h5', 'cnn_model_abc_site_20181208_tensor150_tag90.h5']
    #fnms = ['cnn_model_abc_site_20181130_tensor150.h5']
    #fnms = 'cnn_model_abc_site_20181130_tensor150.h5'
    #for f_cnn in fnms:
    for cnn_f_name in glob.glob(os.path.join(model_rout, '*.h5')):
        print(cnn_f_name)
        tags_list.append(re.findall('tag\d+', cnn_f_name)[0])        
        cnn_classifier_list.append(keras.models.load_model(cnn_f_name))
    
    res = classifier_cnn(text, cnn_classifier_list, tags_list, graph_defaut)
    print(res)
    