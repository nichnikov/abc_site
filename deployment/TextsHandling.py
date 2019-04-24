#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:44:36 2018

@author: alexey
"""
import os, re, subprocess, SplitSentence #, glob, nltk, re, random, shutil
import pandas as pd
  
#функция для лемматизации текстов:
def texts_lemm(text_for_lemmatization):
    p = subprocess.run(['./mystem', '-w', '-c', '-s', '-l'], input = text_for_lemmatization, stdout = subprocess.PIPE, encoding = 'utf-8')
    return p.stdout

def txt_handling(text, lemm_type = 1):
    if lemm_type == 0:
        mask = r'\b\w{0,3}\b'
        shortword = re.compile(mask)
        txt = re.sub('\|','_', text)
        txt = shortword.sub('', txt)
        txt = re.sub('[^\w\_\s]','', txt)
        txt = re.sub('\d+','', txt)
    elif lemm_type == 1:
        txt = re.findall('[аА-яЯ]{3,}', text)
        txt = ' '.join(txt)
    return txt.lower()

def txt_handling_with_lemm(text, lemm_type = 1):
    if lemm_type == 0:
        mask = r'\b\w{0,3}\b'
        shortword = re.compile(mask)
        txt = texts_lemm(text)
        txt = re.sub('\|','_', text)
        txt = shortword.sub('', txt)
        txt = re.sub('[^\w\_\s]','', txt)
        txt = re.sub('\d+','', txt)
        txt = ' '.join(txt)
    elif lemm_type == 1:
        txt = texts_lemm(text)
        txt = re.findall('[аА-яЯ]{3,}', txt)
        txt = ' '.join(txt)
    return txt

#разбиение текста на предложения и лемматизация
#text == текст "как есть" без лемматизации со знаками препинания
def txts_lemm_by_sentences(text):
    txt_by_sen = SplitSentence.SplitSentences(text)
    txt_sens_str = '\n'.join(txt_by_sen)
    lem_txt_sens = texts_lemm(txt_sens_str)
    return [txt_handling(sen) for sen in  lem_txt_sens.split('\n')]#lemm_sens
    
def texts_without_line_break(txts_list):
    #уберем лишние переносы строки:
    return [re.sub('\n', ' ', tx) for tx in txts_list]
    
    #функция, лемматизирующая входящий список текстов (возвращает лемматизированный список того же размера)
    #формат текста на вход: ['добро пожаловать в новый год жопа']
def text_coll_lemmatizer (texts_list):
    txts_for_lemm = '\n'.join([re.sub('\n', ' ', tx) for tx in texts_list])
    lemm_txts = texts_lemm(txts_for_lemm)
    return  [txt_handling(tx) for tx in lemm_txts.split('\n')]



    

#тестирование
if __name__ == '__main__':
    #проверка функции txts_lemm_by_sentences:
    handle_txts_rout = r'/home/alexey/Dropbox/LS_Side_Analysis/data/cases_txts/handle_txts'
    #th = texts_handling_class(lemm_type = 1)
    with open(os.path.join(handle_txts_rout, 'case_2287620.txt'), 'r') as f:
        txt = f.read()
    txt_sens_lemm = txts_lemm_by_sentences(txt)
    print(txt_sens_lemm)
    
    s = txt_sens_lemm[3:6]
    s_ = ' '.join(s).split()
    s_

    '''
    import time
    from collections import defaultdict
    import numpy as np
    from gensim import models
    import pandas as pd

    data_rout = r'/home/alexey/Dropbox/LS_Side_Analysis/data/'
    model_rout = r'/home/alexey/Dropbox/LS_Side_Analysis/models'
    prototype_rout = r'/home/alexey/Dropbox/LS_Side_Analysis/prototype'
    data_cases_txts_rout = r'/home/alexey/Dropbox/LS_Side_Analysis/data/cases_txts/4'
    
    ds1 = pd.read_csv(os.path.join(data_rout, 'data_retail_101ex.csv'))
    ds2 = pd.read_csv(os.path.join(data_rout, 'data_retail_355ex.csv'))
    ds = pd.DataFrame(pd.concat([ds1, ds2], axis = 0))
    #ds = ds.sample(frac=1)
    
    texts_list = list(ds['txt'])
    applicants_list = list(ds['Истец'])
    responders_list = list(ds['Ответчик'])
    tensor_length = 200
    lb = list(ds['тип'])
    
    stoplist = list(set('жопа yanky год рубль суд судно суда судить г копа_коп автомобиль'.split()))   
    #(1) txt_handling
    cl = texts_handling_class(stoplist = [''], lemm_type = 0)
    t = cl.txt_handling_with_lemm("судья на автомобиле судно суда судить судил высудил судебное дело девать постановил свое решение пеня пенить go home yanky 1283")
    print(t)
    cl = texts_handling_class(stoplist = stoplist, lemm_type = 1)
    t = cl.txt_handling_with_lemm("жопа новый год судья на автомобиле судно суда судить судил высудил судебное дело девать постановил свое решение пеня пенить go home yanky 1283")
    print(t)
    
    #(2)text_coll_lemmatizer
    with open (os.path.join(data_cases_txts_rout, '0a0a0f24-48a6-48be-8d23-bc39db756d02.txt'), 'r') as f_tx:
        text = f_tx.read()
    print(text)
    cl = texts_handling_class(stoplist = [''], lemm_type = 0)
    lemm_tx = cl.text_coll_lemmatizer([text])
    print(lemm_tx[0].split()[:50])
    stoplist = ['арбитражный', 'суд', 'свердловский', 'область']
    cl = texts_handling_class(stoplist = stoplist, lemm_type = 1)
    lemm_tx = cl.text_coll_lemmatizer([text])
    lemm_tx
    print(lemm_tx[0].split()[:50])
    
    #(3)text_coll_lemmatizer_replacement    
    cl = texts_handling_class(stoplist = [''], lemm_type = 0)

    lm_txt = cl.text_coll_lemmatizer_replacement(texts_list, applicants_list = applicants_list, responders_list = responders_list)
    print(lm_txt[0])
    
    cl = texts_handling_class(stoplist = [''], lemm_type = 1)
    lm_txt = cl.text_coll_lemmatizer_replacement(texts_list, applicants_list = applicants_list, responders_list = responders_list)
    print(lm_txt[0])
    
    t = cl.txt_handling("судья судно суд постановил свое решение пеня пенить go home yanky 1283")
    print(t)
'''

    