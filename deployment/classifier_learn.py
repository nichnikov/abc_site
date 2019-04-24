#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:40:59 2018

@author: alexey
"""

# LSTM for sequence classification in the IMDB dataset
import os, pickle, TextsHandling, keras
import pandas as pd
import numpy as np

from random import shuffle
from gensim import models, logging

from keras import regularizers
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization


def fragment2tensor (fragment_list, w2v, tensor_length = 150, vec_len = 200):
    vectors_list = w2v[fragment_list[:tensor_length]]    
    if vectors_list.shape[0] < tensor_length:
        add_v = np.array((tensor_length-vectors_list.shape[0])*[[0]*vec_len])
        tensor = np.concatenate((np.array(vectors_list), add_v), axis = 0)
    else:
        tensor = np.array([vectors_list])
    return tensor             

#cnn model creater function
def cnn_model_creator(num_classes):
    model = Sequential()
    model.add(Conv1D(10, kernel_size = 5, strides=1, input_shape=x_train.shape[1:], kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=1))
    model.add(Flatten())
    model.add(Dense(300, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])    
    return model

# create the model
def lstm_model_creater(num_classes):
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(500, dropout = 0.1, recurrent_dropout=0.1))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#model_rout = r'/home/alexey/Dropbox/abc_site/models'
#data_rout = r'/home/alexey/Dropbox/abc_site/data/'
#dict_rout = r'/home/alexey/Dropbox/abc_site/dicts'

model_rout = r'/home/an/Dropbox/abc_site/models'
data_rout = r'/home/an/Dropbox/abc_site/data/'
dict_rout = r'/home/an/Dropbox/abc_site/dicts'


#лемматизация текстов:
ds = pd.read_csv(os.path.join(data_rout, 'vtiger_troubletickets.csv'), header=None)
ds.rename(columns={0:'texts', 1:'tags'}, inplace=True)

texts_for_lemm = list(ds['texts'])
tags = [tg for tg in ds['tags'].str.split(',')]

lemm_texts = TextsHandling.text_coll_lemmatizer(texts_for_lemm)

#for tag = 22:
tags_with22_nums = [num for num, tag in enumerate(tags) if '22' in tag]
tags_no22_nums = [num for num, tag in enumerate(tags) if num not in tags_with22_nums]

#нужно сделать word2vec модель
w2v_model = models.Word2Vec([tx.split() for tx in lemm_texts], min_count=5, iter=15, size=150, window=5,  workers=7)

work_vocab = [w for w in w2v_model.wv.vocab]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
w2v_model.save(os.path.join(model_rout, 'w2v_model_abc_tech_20181130'))

lemm_texts_true_words = [[w for w in sen.split() if w in work_vocab] for sen in lemm_texts]
w2v = models.Word2Vec.load(os.path.join(model_rout, 'w2v_model_abc_tech_20181130'))

nums_frs_lb0 = [num_fr for num_fr in enumerate(lemm_texts_true_words) if num_fr[0] in tags_no22_nums]
nums_frs_lb1 = [num_fr for num_fr in enumerate(lemm_texts_true_words) if num_fr[0] in tags_with22_nums]
#сформируем тензоры для обучения сверточной нейронной сети:
tensors_lb0 = np.zeros((1, 150, 150))
for num_fr in nums_frs_lb0:
    try:
        ts = fragment2tensor(num_fr[1], w2v, tensor_length = 150, vec_len = 150)
        tensors_lb0 = np.concatenate((tensors_lb0, ts.reshape(1, 150, 150)), axis=0)
    except:
        None

tensors_lb1 = np.zeros((1, 150, 150))
for num_fr in nums_frs_lb1:
    try:
        ts = fragment2tensor(num_fr[1], w2v, tensor_length = 150, vec_len = 150)
        tensors_lb1 = np.concatenate((tensors_lb1, ts.reshape(1, 150, 150)), axis=0)
    except:
        None

tensors_lb0 = np.delete(tensors_lb0, 0, axis=0)
tensors_lb1 = np.delete(tensors_lb1, 0, axis=0)

tensors_lb0 = list(zip(tensors_lb0, tensors_lb0.shape[0]*[0]))
tensors_lb1 = list(zip(tensors_lb1, tensors_lb1.shape[0]*[1]))

tensors_lb = tensors_lb0 + tensors_lb1

shuffle(tensors_lb)

dataset_sh_x, lb_sh = list(zip(*tensors_lb))
x_train = np.array(dataset_sh_x)
y_train = np.array(lb_sh)

#обучение сверточной сети:
batch_size = 32
num_classes = 2
epochs = 150
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cnn_model_abc_site_20181130_tensor150.h5'

# The data, split between train and test sets:
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')


# Convert class vectors to binary class matrices.
x_train = x_train.astype('float32') #x_train.reshape(4000,200, 60).astype('float32')
y_train = keras.utils.to_categorical(y_train, num_classes)
print (x_train.shape)

model = cnn_model_creator(num_classes)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)

#Save model and weights
#if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)

model_path = os.path.join(model_rout, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


'''
tokenizer = Tokenizer(num_words=500, oov_token=None)
tokenizer.fit_on_texts(txts_for_learning)

sec = tokenizer.texts_to_sequences(txts_for_learning)

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
num_classes = 2
data_augmentation = True
top_words = 500
model_name = 'lstm_model_vector100_20181026_var2.h5'

#sequences = tokenizer.texts_to_sequences(X)

# truncate and pad input sequences
max_review_length = 100
X_train = sequence.pad_sequences(sec[:340], maxlen=max_review_length)
y_train = keras.utils.to_categorical(np.array(lbs_for_learning[:340]), num_classes)

X_val = sequence.pad_sequences(sec[340:], maxlen=max_review_length)
Y_val = keras.utils.to_categorical(np.array(lbs_for_learning[340:]), num_classes)

print(model.summary())
model = lstm_model_creater(num_classes)
model.fit(X_train, y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=64)

model_path = os.path.join(model_rout, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

with open(os.path.join(model_rout, 'lstm_tokenizer_20181026.pickle'), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
