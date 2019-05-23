#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:32:04 2019

@author: alexey
"""
import os, pickle
import pandas as pd
from functions import new_nn_learning

model_rout = r'./models'
data_rout = r'./data/'
dict_rout = r'./dicts'

#выгрузим размеченные тексты
ds = pd.DataFrame(pd.read_csv(os.path.join(data_rout, 'txts_lbls.csv')))
#print(ds[:100])
print(ds[ds['lbs']==6])

#обучим нейронные сети по тем датасетам, которые для этого необходимы
new_nn_learning(model_rout, data_rout, dict_rout, ds, min_class_examples=100)