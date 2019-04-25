#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:44:36 2018

@author: alexey
"""
import re, subprocess, SplitSentence#, os, glob, nltk, re, random, shutil
import nltk, nltk.data
import nltk.tokenize.punkt

#функция для лемматизации текстов:
def texts_lemm(text_for_lemmatization):
    p = subprocess.run(['./mystem', '-w', '-c', '-s', '-l'], input = text_for_lemmatization, stdout = subprocess.PIPE, encoding = 'utf-8')
    return p.stdout

def txt_handling(text, lemm_type = 1):
    if lemm_type == 0:
        mask = r'\b\w{0,1}\b'
        shortword = re.compile(mask)
        txt = re.sub(r'\|','_', text)
        txt = shortword.sub('', txt)
        txt = re.sub(r'[^\w\_\s]','', txt)
        txt = re.sub(r'\d+','', txt)
    elif lemm_type == 1:
        txt = re.findall('[аА-яЯ]{1,}', text)
        txt = ' '.join(txt)
    return txt.lower()

def txt_handling_with_lemm(text, lemm_type = 1):
    if lemm_type == 0:
        mask = r'\b\w{0,1}\b'
        shortword = re.compile(mask)
        txt = texts_lemm(text)
        txt = re.sub(r'\|','_', text)
        txt = shortword.sub('', txt)
        txt = re.sub(r'[^\w\_\s]','', txt)
        txt = re.sub(r'\d+','', txt)
        txt = ' '.join(txt)
    elif lemm_type == 1:
        txt = texts_lemm(text)
        txt = re.findall('[аА-яЯ]{1,}', txt)
        txt = ' '.join(txt)
    return txt

#разбиение текста на предложения и лемматизация
#text == текст "как есть" без лемматизации со знаками препинания
def txts_lemm_by_sentences(text, lemm_type = 1):    
    txt_by_sen = SplitSentence.SplitSentences(text)
    txt_sens_str = '\n'.join(txt_by_sen)
    lem_txt_sens = texts_lemm(txt_sens_str)
    lemm_sens = [txt_handling(sen, lemm_type) for sen in lem_txt_sens.split('\n')]
    del lemm_sens[-1:]
    return lemm_sens

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
    lemm_txts = texts_lemm(txts_for_lemm)
    result = [txt_handling(tx, lemm_type) for tx in lemm_txts.split('\n')]
    del result[-1:]
    return result

#тестирование
if __name__ == '__main__':
    #проверка функции txts_lemm_by_sentences:  
    txt = 'постой паровоз, Не стучите, колеса, Кондуктор нажмИ на Тормоза, я К маменьке роднйо с последним приветом спешу показаться на глаза'
    txt_sens_lemm = txts_lemm_by_sentences(txt, lemm_type = 0)
    print(txt_sens_lemm)
    
    tx_lm_coll = text_coll_lemmatizer([txt], 0)
    print(tx_lm_coll)
    
    text_sens = 'Только знайте, любезная Катерина Матвевна, что классовые сражения на сегодняшний день в общем и целом завершены, и час всемирного освобождения настает. И пришел мне черед домой возвратиться, чтобы с вами вместе строить новую жизнь в милой сердцу родной стороне.'
    print(txts_lemm_by_sentences(text_sens, lemm_type = 0))