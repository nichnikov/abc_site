#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:13:33 2019
анализируем входящий csv файл с полями "text, lbs", решаем, сколько сетей нужно построить
создает файл с классами для данного проекта

@author: alexey
"""
import os, re, pickle
import pandas as pd
from IterableTexstsHandling import IndexingTextsList

model_rout = r'./models'
data_rout = r'./data/'
dict_rout = r'./dicts'


ds = pd.DataFrame(pd.read_csv(os.path.join(data_rout, 'txts_lbls.csv')))

class_q = pd.DataFrame(ds['txt'].groupby(ds['lbs']).count())
class_q.reset_index(inplace = True)
class_for_learning = class_q[class_q['txt'] >=100]

num_classes = class_for_learning.shape[0]
ds_lbs_for_learn = ds.loc[ds['lbs'].isin(list(class_for_learning['lbs']))]

lbs = list(ds_lbs_for_learn['lbs'])
txts = list(ds_lbs_for_learn['txt'])

#предобработка текстов:
#заменим ссылки
txts_ = [re.sub(r'www\.\w+\.\w+|\w+\.\w+\.рф', 'ссылканасайт ', tx) for tx in txts]
txts_ = [re.sub('[aA-zZ]', ' ', tx) for tx in txts_]
txts_ = [re.sub(r'[^\w\s]', '', tx) for tx in txts_]
txts_ = [re.sub(r'[\d+]', '', tx) for tx in txts_]
txts_ = [re.sub(r'\s+', ' ', tx) for tx in txts_]

txt_inx = IndexingTextsList(enumerate(txts_))
txt_inx.lemmatization(inplace=True, lemm_type=1)

with open(os.path.join(data_rout, 'lebled_texts.pickle'), 'wb') as f:
    pickle.dump(list(zip(txt_inx.lemm_texts, lbs)), f)