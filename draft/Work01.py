#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:32:04 2019

@author: alexey
"""

import os
import pandas as pd
from functions import text_coll_lemmatizer, data_for_learning_selection, texts_prepare, tag_handling

model_rout = r'./models'
data_rout = r'./data/'
dict_rout = r'./dicts'

#выгрузим размеченные тексты
ds = pd.DataFrame(pd.read_csv(os.path.join(data_rout, 'txts_lbls.csv')))

#отберем те классы примеров, в которых больше n (например, n=100) текстов:
txts, lbls = zip(*data_for_learning_selection(ds, each_class_examples=100))


#проведем некоторые замены (перечисляются в списке кортежей):
patterns = [(r'www\.\w+\.\w+|\w+\.\w+\.рф', 'ссылканасайт '), ('[aA-zZ]', ' '),
            (r'[^\w\s]', ''), (r'[\d+]', ''), (r'\s+', ' ')]

txts_ = texts_prepare(txts, patterns)

#лемматизация текстов:
lemm_txts = text_coll_lemmatizer(txts_)

#выбор одного тега для работы с ним (остальные тексты считаются классом 0) с балансировкой или без
#применимо к классификации "один ко многим"
data = list(zip(lemm_txts, lbls))
txs0, txs1 = tag_handling(data, 22, balance = True)

#создание word2vec модели
from gensim import models
w2v_model = models.Word2Vec([tx.split() for tx in lemm_txts], min_count=30, iter=25, size=150, window=5,  workers=7)
w2v_model.wv.most_similar('ссылканасайт')

#создание w2v модели
work_vocab = [w for w in w2v_model.wv.vocab]
w2v_model.save(os.path.join(model_rout, 'w2v_model_abc_tech_20190214'))


txts, lbls = zip(*data)

lemm_texts_true_words = [[w for w in sen.split() if w in work_vocab] for sen in txts]
w2v = models.Word2Vec.load(os.path.join(model_rout, 'w2v_model_abc_tech_20190214'))
w2v.wv.most_similar('ссылканасайт')



print(txs0[:5])
print(txs1[:5])
print(len(txs0), len(txs1))