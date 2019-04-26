#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:13:33 2019
анализируем входящий csv файл с полями "text, lbs", решаем, сколько сетей нужно построить
создает файл с классами для данного проекта

@author: alexey
"""
import os
import pandas as pd
from functions import text_coll_lemmatizer, data_for_learning_selection, texts_prepare

model_rout = r'./models'
data_rout = r'./data/'
dict_rout = r'./dicts'


ds = pd.DataFrame(pd.read_csv(os.path.join(data_rout, 'txts_lbls.csv')))

#предобработка текстов:
#заменим ссылки
txts, lbls = zip(*data_for_learning_selection(ds, each_class_examples=100))

patterns = [(r'www\.\w+\.\w+|\w+\.\w+\.рф', 'ссылканасайт '), ('[aA-zZ]', ' '),
            (r'[^\w\s]', ''), (r'[\d+]', ''), (r'\s+', ' ')]

txts_ = texts_prepare(txts, patterns)

#лемматизация текстов
lemm_txts = text_coll_lemmatizer(txts_)
print(lemm_txts[:10])

