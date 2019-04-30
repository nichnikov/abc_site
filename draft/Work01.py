#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:32:04 2019

@author: alexey
"""

import os, pickle
import pandas as pd
from functions import text_coll_lemmatizer, data_for_learning_selection, texts_change_patterns, tag_handling
import keras
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers.core import SpatialDropout1D
from keras import regularizers
import numpy as np
from random import shuffle
from gensim import models

def make_embedded_sequences(tokenizer, w2v_model, MAX_SEQUENCE_LENGTH = 15):
    embeddings_index = {}
    
    #tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
    for word in tokenizer.word_index:
        embeddings_index[word] = w2v_model.wv[word]
    
    w2v_model.wv[word]
    #embeddings_index['просьба']
    #embeddings_index.get('просьба')
    EMBEDDING_DIM = embeddings_index.get(work_vocab[0]).shape[0]
    
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
    embedding_matrix.shape
        
    for word, i in tokenizer.word_index.items():    
        embedding_vector = embeddings_index.get(word)
        embedding_matrix[i] = embedding_vector
    
    nb_words = len(tokenizer.word_index) + 1
    WV_DIM = EMBEDDING_DIM
    wv_matrix = embedding_matrix
    wv_layer = Embedding(nb_words,
                         WV_DIM,
                         mask_zero=False,
                         weights=[wv_matrix],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=False)

    # Inputs
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedded_sequences = wv_layer(comment_input)                
    # biGRU
    embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
    #x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(embedded_sequences)
    #x = LSTM(100)(embedded_sequences)
    return embedded_sequences, comment_input
    
def make_lstm_nn(embedded_sequences, comment_input, num_classes):
    x = LSTM(1000, dropout = 0.1, activation='tanh', recurrent_dropout=0.1, 
             kernel_regularizer=regularizers.l2(0.02), return_sequences=False)(embedded_sequences)
    
    # Output
    x = Dense(500)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    preds = Dense(num_classes, activation='sigmoid')(x)#num_classes
        
    # build the model
    model = Model(inputs=[comment_input], outputs=preds)
    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.SGD(lr=0.001, 
                                                 momentum=0.09, nesterov=True), metrics=['accuracy'])
    return model

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

txts_ = texts_change_patterns(txts, patterns)

#лемматизация текстов:
lemm_txts = text_coll_lemmatizer(txts_)

#создание word2vec модели
w2v_model = models.Word2Vec([tx.split() for tx in lemm_txts], 
                             min_count=30, iter=25, size=150, window=5,  workers=7)

#проверка:
#w2v_model.wv.most_similar('ссылканасайт')

#сохранение w2v модели
work_vocab = [w for w in w2v_model.wv.vocab]
w2v_model.save(os.path.join(model_rout, 'w2v_model'))

texts_for_working = [[w for w in sen.split() if w in work_vocab] for sen in lemm_txts]

#определим токенайзер:
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_for_working)

#сохранение токенезатора модели
with open(os.path.join(model_rout, 'tokenizer.pickle'), 'wb') as file:
    pickle.dump(tokenizer, file)


#выбор одного тега для работы с ним (остальные тексты считаются классом 0) с балансировкой или без
#применимо к классификации "один ко многим"
texts_with_lables = list(zip(texts_for_working, lbls))

#блок сохранения и загрузки моделей:
'''
w2v_model = models.Word2Vec.load(os.path.join(model_rout, 'w2v_model_abc_tech_20190214'))
print(w2v_model.wv.most_similar('ссылканасайт'))

'''

for tag_num in list(set(lbls)):
    txs0, txs1 = tag_handling(texts_with_lables, tag_num, balance = True)
    
    texts_for_learning = txs0 + txs1
    lbs = len(txs0)*[0] + len(txs1)*[1]        
    print('Создаем embedding_matrix')
    
    #подготовка данных для обучения моделей:
    labled_texts = list(zip(texts_for_learning, lbs))
    shuffle(labled_texts)
    txts_for_learning, lbs_for_learning = zip(*labled_texts)
    
    MAX_SEQUENCE_LENGTH = 15
    embedded_sequences, comment_input = make_embedded_sequences(tokenizer, w2v_model, MAX_SEQUENCE_LENGTH)
    
    num_classes = len(set(lbs_for_learning))    
    model = make_lstm_nn(embedded_sequences, comment_input, num_classes)
    
    # pad
    sequences = [[tokenizer.word_index.get(t, 0) for t in comment] for comment in txts_for_learning]    
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
    y = keras.utils.to_categorical(np.array(lbs_for_learning), num_classes)
    history = model.fit([data], y, validation_split=0.2, epochs=5, batch_size=32, shuffle=True)
        
    import matplotlib.pyplot as plt
    
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    model_name = str(tag_num) + '_lstm_model.h5'
    model.save(os.path.join(model_rout, model_name))  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model
