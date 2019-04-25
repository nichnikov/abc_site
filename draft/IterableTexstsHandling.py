#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:26:53 2019

@author: alexey
"""
import TextsHandling, nltk

def flatten_all (iterable):
	for elem in iterable:
		if not isinstance(elem, list):
			yield elem
		else:
			for x in flatten_all(elem):
				yield x

class IndexingTextsList(object):
    def __init__(self, list_of_tuples_with_indexes_texts):
        self.indexes, self.textslist = zip(*list_of_tuples_with_indexes_texts)
        self.lemm_texts = []
    
    #удаляет тексты из списка без потери нумерации
    def remove_txts(self, remove_txts_list, inplace = False):
        list_tx_removed_data = []                 
        for tx in self.textslist:
            if tx in remove_txts_list:
                list_tx_removed_data.append('')
            else:
                list_tx_removed_data.append(tx)
        if inplace == True:
            self.textslist = list_tx_removed_data
        return list(zip(self.indexes, list_tx_removed_data))
    
    #выбор элементов по значению
    def select_txts(self, select_txts_list):
        return [num_tx for num_tx in list(zip(self.indexes, self.textslist)) if  num_tx[1] in select_txts_list]
 
    #выбор элементов по индексам
    def select_txts_by_indexs(self, select_indexs_list):
        return [num_tx for num_tx in list(zip(self.indexes, self.textslist)) if  num_tx[0] in select_indexs_list]
    
    #разбивка на предложения входящих данных
    def split_sentences(self, inplace = False):
        sens_list = [nltk.sent_tokenize(tx, language="russian") for tx in [tx for tx in flatten_all(self.textslist)]]
        if inplace == True:
            new_indexes = []
            texts_by_sens = []
            for indx, tx_list in zip(self.indexes, sens_list):
                new_indexes.append([indx]*len(tx_list))
                for sen in flatten_all(tx_list):
                    texts_by_sens.append(sen)
            self.textslist = texts_by_sens
            self.indexes = [inx for inx in flatten_all(new_indexes)]
            result = list(zip(self.indexes, self.textslist))
        else:
            result = list(zip(self.indexes, sens_list))
        return result
    
    #лемматизация, в случае, если by_sentences = True и inplace = True - разбивает входящий текст на предложеняи и для каждого предложения проставляет индекс того 
    #элемента, к которому относился текст из которого получено предложение
    def lemmatization(self, by_sentences = False, lemm_type = 1, inplace = False):
        if by_sentences == False:
            #print(self.textslist)
            self.lemm_texts = TextsHandling.text_coll_lemmatizer([tx for tx in flatten_all(self.textslist)], lemm_type = lemm_type)            
            #print(TextsHandling.text_coll_lemmatizer([tx for tx in flatten_all(self.textslist)], lemm_type = lemm_type))
            result = list(zip(self.indexes, self.lemm_texts))
        elif by_sentences == True:            
            if inplace == True:
                self.split_sentences(inplace = True)
                self.lemm_texts = [lemm_tx for lemm_tx in flatten_all([TextsHandling.text_coll_lemmatizer(self.textslist, lemm_type = lemm_type)])]
                result = list(zip(self.indexes, self.lemm_texts))
            elif inplace == False:
                txts_lemm = [TextsHandling.text_coll_lemmatizer(txt_by_sens, lemm_type = lemm_type) for indx, txt_by_sens in self.split_sentences()]
                result = list(zip(self.indexes, txts_lemm))
        return result

    
if __name__ == '__main__':
    txt = ['привет, пока', 'я!', 'добрый день', 'как жизнь', 'здравствуй и прощай', 'мы потратили время', 'сколько стоит дом построить', 'как жизнь молодая']
    indxs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']
    inds_txts = list(zip(indxs, txt))    
    intxts = IndexingTextsList(inds_txts)
    print(intxts.textslist)
    print(intxts.indexes)
    print(intxts.remove_txts(['я!', 'как жизнь']))
    #print(intxts.remove_txts(['я!', 'как жизнь'], inplace = True))
    print(intxts.select_txts(['я!', 'добрый день']))
    print(intxts.select_txts_by_indexs(['a1', 'a6']))
    print(intxts.textslist)
    print(intxts.lemmatization(lemm_type=0))    
    
    sens_txts = ['Клара у Карла украла кораллы. Сколько веревочке не виться. Как вы там поживаете, дорогая Марфа Сергеевна?', 'Раису Ивановну хочу. Наши летчики самые летные летчики в Мире!', 'Я да ты да мы.']
    indxs_sens_txts = list(zip(['b1', 'b2', 'b3'], sens_txts))
    int_sens = IndexingTextsList(indxs_sens_txts)
    print(int_sens.split_sentences())
    print(int_sens.split_sentences(inplace = True))

    print(int_sens.textslist)
    print(int_sens.indexes)
    print(int_sens.lemmatization())
    print(int_sens.lemmatization(by_sentences=False, lemm_type=0))
    #print(int_sens.lemmatization(by_sentences=True, lemm_type=0))
    print(int_sens.lemmatization(by_sentences=True, lemm_type=1))
    
    test = int_sens.lemmatization(by_sentences=True, lemm_type=1)
    a, b = zip(*test)
    print(a)
    print(b)
    print(int_sens.lemmatization(by_sentences=True, lemm_type=1, inplace = True))
    print(int_sens.textslist)
    print(int_sens.lemm_texts)
    print(int_sens.indexes)
    print(intxts.lemm_texts)
    print(intxts.indexes)
    print(list(zip(intxts.indexes, intxts.textslist, intxts.lemm_texts)))
    print(list(zip(int_sens.indexes, int_sens.textslist, int_sens.lemm_texts)))
