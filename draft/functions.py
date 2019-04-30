#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:09:51 2019

@author: alexey
"""

import os, re, copy, time
from pymystem3 import Mystem
import pandas as pd
from random import shuffle
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import models
from gensim import similarities
from multiprocessing import Pool
from functools import partial

def texts_lemm_mystem(text_for_lemmatization):
    mystem = Mystem()
    return ' '.join(mystem.lemmatize(text_for_lemmatization))

#замена паттернов (формат кортежа)
def texts_change_patterns(txts, patterns):
    for t, p in patterns:
        txts = [re.sub(t, p, tx) for tx in txts]   
    return txts

patterns = [(r'\|','_'), (r'[^\w\_\s]',''), (r'\d+',''), (r'\b\w{0,2}\b', '')]

def txt_handling(text, lemm_type = 1):
    if lemm_type == 0:
        txt = texts_change_patterns(text, patterns)
    elif lemm_type == 1:
        txt = re.findall('[аА-яЯ]{3,}', text)
        txt = ' '.join(txt)
    return txt.lower()

def texts_without_line_break(txts_list):
    #уберем лишние переносы строки:
    txt_ = []
    for tx in txts_list:
        txt_.append(re.sub('\n', ' ', tx))    
    return txt_

#функция, лемматизирующая входящий список текстов (возвращает лемматизированный список того же размера)
#формат текста на вход: ['добро пожаловать в новый год жопа']
def text_coll_lemmatizer (texts_list, lemm_type = 1):
    #для лемматизации объединим тексты из списка в один с разделителем (перенос строки):
    txts_for_lemm = '\n'.join(texts_without_line_break(texts_list))
    #lemm_txts = texts_lemm(txts_for_lemm)
    lemm_txts = texts_lemm_mystem(txts_for_lemm)
    result = [txt_handling(tx, lemm_type) for tx in lemm_txts.split('\n')]
    del result[-1:]
    return result

def SliceArray(src:[], length:int=1, stride:int=1) -> [[]]:
    return [src[i:i+length] for i in range(0, len(src), stride)]

def flatten_all(iterable):
	for elem in iterable:
		if not isinstance(elem, list):
			yield elem
		else:
			for x in flatten_all(elem):
				yield x
                
#отбор классов для обучения (классов, в которых примеров больше, чем n)
def data_for_learning_selection(ds, each_class_examples=100):
    class_q = pd.DataFrame(ds['txt'].groupby(ds['lbs']).count())
    class_q.reset_index(inplace = True)
    class_for_learning = class_q[class_q['txt'] >=each_class_examples]
    
    ds_lbs_for_learn = ds.loc[ds['lbs'].isin(list(class_for_learning['lbs']))]
    
    return list(zip(list(ds_lbs_for_learn['txt']), list(ds_lbs_for_learn['lbs'])))

#выбор одного тега для работы с ним (остальные тексты считаются классом 0) с балансировкой или без
#применимо к классификации "один ко многим"
def tag_handling(labled_txts_list, teg_num, balance = False):
    #отберем тексты, с заданным тегом и тексты без этого тега
    txts_with_tag = [x[0] for x in labled_txts_list if x[1] == teg_num]
    txts_without_tag = [x[0] for x in labled_txts_list if x[1] != teg_num]
    #сбалансируем количество текстов с тегом и без    
    if balance == True:
        shuffle(txts_without_tag)
        return txts_with_tag, txts_without_tag[:len(txts_with_tag)]
    else:
        return txts_with_tag, txts_without_tag